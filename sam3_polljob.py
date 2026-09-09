#!/usr/bin/env python3
"""Bounded, read-only SAM3 MCP job polling (standard library only)."""
import argparse
import json
from pathlib import Path
import queue
import threading
import time
import urllib.error
import urllib.request


class PollError(Exception):
    """An error that retrying the same request cannot repair."""


def decode_reply(reply, request_id):
    if not isinstance(reply, dict) or reply.get('id') != request_id:
        raise PollError('Unexpected MCP response ID or response format')
    if 'error' in reply:
        raise PollError(f"MCP error: {reply['error']}")
    result = reply.get('result', {})
    if result.get('isError'):
        raise PollError(f"MCP tool error: {result.get('content')}")
    state = result.get('structuredContent')
    if state is None:
        blocks = result.get('content', [])
        text = next((block['text'] for block in blocks if block.get('type') == 'text'), None)
        if text is None:
            raise PollError('MCP response has no job status')
        state = json.loads(text)
    if not isinstance(state, dict) or state.get('status') not in ('queued', 'running', 'completed', 'failed'):
        raise PollError('MCP response has an invalid job status')
    return state


def read_reply(response, request_id):
    if 'text/event-stream' not in response.headers.get('Content-Type', ''):
        return decode_reply(json.load(response), request_id)
    # SSE connections may stay open after sending the answer. Stop at the matching
    # event, rather than resp.read() waiting for the whole connection to close.
    data = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                reply = json.loads('\n'.join(data))
                data = []
                if isinstance(reply, dict) and reply.get('id') == request_id:
                    return decode_reply(reply, request_id)
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
    if data:
        return decode_reply(json.loads('\n'.join(data)), request_id)
    raise PollError('MCP stream ended without a matching response')


def fetch_status(url, session_id, job_id, timeout, request_id):
    body = dict(jsonrpc='2.0', id=request_id, method='tools/call',
                params=dict(name='get_job_status', arguments=dict(job_id=job_id)))
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream',
        'mcp-session-id': session_id})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return read_reply(response, request_id)
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500 and exc.code not in (408, 429):
            raise PollError(f'HTTP {exc.code}: check the MCP URL/session; an expired session must be renewed') from exc
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise PollError(f'Malformed MCP response: {exc}') from exc


def bounded_call(function, timeout):
    """Cap the whole request, even if an SSE heartbeat keeps the socket alive."""
    output = queue.Queue(maxsize=1)
    def run():
        try:
            output.put((True, function()))
        except Exception as exc:
            output.put((False, exc))
    threading.Thread(target=run, daemon=True, name='sam3-status-request').start()
    try:
        ok, value = output.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f'HTTP request exceeded {timeout:g}s total deadline') from None
    if not ok:
        raise value
    return value


def poll(fetch, *, interval=20, request_timeout=15, max_errors=3, max_wait=1800,
         stall_timeout=600, clock=time.monotonic, sleep=time.sleep,
         emit=lambda message: print(message, flush=True)):
    started = last_progress = clock()
    previous = None
    errors = request_id = 0
    while True:
        now = clock()
        elapsed = now-started
        if elapsed >= max_wait:
            emit(f'STOP: reached total wait limit ({elapsed:.1f}s). Server job was not cancelled.')
            return 2
        if now-last_progress >= stall_timeout:
            emit(f'STOP: no observed progress for {now-last_progress:.1f}s. Check server logs; job was not cancelled.')
            return 2
        request_id += 1
        try:
            timeout = min(request_timeout, max_wait-elapsed, stall_timeout-(now-last_progress))
            state = fetch(timeout, request_id)
            now = clock()
            status, internal = state.get('status'), state.get('internal_status')
            signature = (status, internal, state.get('attempt'), state.get('processed_frames'))
            if signature != previous:
                last_progress, previous = now, signature
            errors = 0
            emit(f"[{now-started:.1f}s] status={status} internal={internal} attempt={state.get('attempt')} "
                 f"frames={state.get('processed_frames')}/{state.get('total_frames')}")
            if internal == 'awaiting_keyframe_review':
                emit(f"ACTION: inspect keyframe previews and submit review; pending={state.get('pending_keyframes')}")
                return 0
            if internal == 'awaiting_visual_review':
                emit('ACTION: tracking and QA completed; inspect final previews and submit visual review.')
                return 0
            if status == 'failed':
                emit(f"FAILED: {state.get('error')}")
                return 1
            if status == 'completed':
                emit(f'FINISHED: internal_status={internal}; no further polling needed.')
                return 0
            if status not in ('queued', 'running'):
                raise PollError(f'Unknown job status: {status!r}')
        except PollError as exc:
            emit(f'ERROR: {exc}; stopping instead of retrying indefinitely.')
            return 1
        except (OSError, TimeoutError) as exc:
            errors += 1
            emit(f'[{clock()-started:.1f}s] HTTP error {errors}/{max_errors}: {exc}')
            if errors >= max_errors:
                emit('STOP: consecutive HTTP failures reached limit. Server job was not cancelled.')
                return 2
        delay = min(interval, max_wait-(clock()-started), stall_timeout-(clock()-last_progress))
        if delay > 0:
            sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('interval', type=float, nargs='?', default=20)
    parser.add_argument('--url', default='http://100.120.152.79:8765/mcp')
    parser.add_argument('--session-file', type=Path, default=Path('/tmp/sam3_mcp_sid.txt'))
    parser.add_argument('--request-timeout', type=float, default=15)
    parser.add_argument('--max-errors', type=int, default=3)
    parser.add_argument('--max-wait', type=float, default=1800)
    parser.add_argument('--stall-timeout', type=float, default=600)
    args = parser.parse_args()
    if min(args.interval, args.request_timeout, args.max_errors, args.max_wait, args.stall_timeout) <= 0:
        parser.error('All intervals, timeouts and limits must be positive')
    try:
        session_id = args.session_file.read_text().strip()
        if not session_id:
            raise PollError('MCP session file is empty')
        def fetch(timeout, request_id):
            return bounded_call(lambda: fetch_status(args.url, session_id, args.job_id, timeout, request_id), timeout)
        return poll(fetch, interval=args.interval, request_timeout=args.request_timeout,
                    max_errors=args.max_errors, max_wait=args.max_wait, stall_timeout=args.stall_timeout)
    except (OSError, PollError) as exc:
        print(f'ERROR: {exc}', flush=True)
        return 1
    except KeyboardInterrupt:
        print('Polling interrupted; server job was not cancelled.', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
