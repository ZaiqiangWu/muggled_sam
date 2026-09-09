#!/usr/bin/env python3
"""Single-GPU SAM3 MCP server. Run inside the repository's SAM3 environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import copy
import json
import logging
from pathlib import Path
import shutil
import tarfile
import threading
import time
import traceback
import uuid

from sam3_http_files import DEFAULT_MAX_UPLOAD_BYTES, download_reference, register_file_routes
from sam3_mcp_pipeline import SAM3Backend, analyze, video_frame_count, write_json

LOG = logging.getLogger('sam3-mcp')


class Service:
    def __init__(self, backend, root, input_roots):
        self.backend = backend
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.input_roots = [Path(p).resolve() for p in input_roots]
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sam3-gpu')
        self.jobs = {}
        for path in self.root.rglob('job.json'):
            try:
                job = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                LOG.warning('Skipping unreadable job manifest %s: %s', path, exc)
                continue
            if not isinstance(job, dict) or not isinstance(job.get('job_id'), str) or not job.get('status'):
                LOG.warning('Skipping malformed job manifest %s', path)
                continue
            if job['status'] in ('queued','running'):
                job.update(status='failed', error='Server restarted during inference; submit a retry')
            for attempt in job.get('attempts',[]):
                if isinstance(attempt, dict) and attempt.get('status') in ('queued','running'):
                    attempt.update(status='failed', error=job['error'])
            self.jobs[job['job_id']] = job

    def save(self, job):
        write_json(Path(job['directory'])/'job.json', job)

    def get(self, job_id):
        if job_id not in self.jobs:
            raise ValueError('Unknown job_id')
        return self.jobs[job_id]

    def resolve_video(self, video_path):
        """Resolve video_path against the configured input roots.

        Accepts the relative path returned by POST /upload (e.g. \"abc123/video.mp4\")
        or an absolute path inside a --input-root; relative names are resolved against
        every input root and must exist in exactly one of them.
        """
        raw = Path(video_path).expanduser()
        candidates = [raw] if raw.is_absolute() else [root/raw for root in self.input_roots]
        resolved = [c for c in (candidate.resolve() for candidate in candidates)
                    if c.is_file() and any(c.is_relative_to(root) for root in self.input_roots)]
        if not resolved:
            raise ValueError('Input must be an existing file under a configured --input-root; '
                             'upload Mac videos with POST /upload and pass the returned relative path')
        if len(set(resolved)) > 1:
            raise ValueError('Relative video_path exists under multiple --input-root entries; use an absolute path')
        return resolved[0]

    def result(self, job_id):
        with self.lock:
            result = copy.deepcopy(self.get(job_id))
        result['next_step'] = ('Call get_preview for every preview frame, then submit_visual_review; '
                               'if rejected, call rerun_segmentation with corrected parameters.')
        if result['status'] == 'awaiting_keyframe_review':
            result['next_step'] = ('Call get_keyframe_preview then submit_keyframe_review for every keyframe '
                                   'in the latest attempt. Tracking starts only after all pass.')
        elif result['status'] in ('queued', 'running'):
            result['next_step'] = ('Poll get_job_status; awaiting_keyframe_review requires get_keyframe_preview '
                                   'and submit_keyframe_review for every pending keyframe.')
        elif result['status'] in ('needs_retry', 'retry_exhausted'):
            result['next_step'] = ('Call rerun_segmentation with corrected parameters.' if
                                   result['status'] == 'needs_retry' else 'Retry budget exhausted.')
        result['outputs'] = self.output_paths(result)
        return result

    def output_paths(self, job):
        """Work-root-relative artifact paths for this job, usable as GET /download/<path>."""
        directory = Path(job['directory'])
        outputs = dict(job_json=(directory/'job.json').relative_to(self.root).as_posix())
        best = job.get('best_attempt')
        if best is not None and best < len(job['attempts']):
            attempt = job['attempts'][best]
            if 'rgba_dir' in attempt:
                attempt_dir = Path(attempt['directory'])
                outputs.update(
                    rgba_dir=Path(attempt['rgba_dir']).relative_to(self.root).as_posix(),
                    validation_video=(attempt_dir/'overlay.mp4').relative_to(self.root).as_posix(),
                    analysis=(attempt_dir/'analysis.json').relative_to(self.root).as_posix(),
                    previews={str(frame): (attempt_dir/'previews'/f'{frame:08d}.png').relative_to(self.root).as_posix()
                              for frame in attempt.get('preview_frames', [])})
        return outputs

    @staticmethod
    def public_status(status):
        """Map internal states to the four coarse job states for polling."""
        return {'queued':'queued','running':'running','failed':'failed',
                'awaiting_keyframe_review':'running'}.get(status,'completed')

    def job_status(self, job_id):
        with self.lock:
            job = copy.deepcopy(self.get(job_id))
        attempt = job['attempts'][-1] if job['attempts'] else {}
        total = attempt.get('total_frames')
        processed = attempt.get('processed_frames')
        state = dict(job_id=job_id, status=self.public_status(job['status']),
                     internal_status=job['status'], attempt=attempt.get('index'),
                     processed_frames=processed, total_frames=total,
                     progress=round(processed/total,4) if total and processed is not None else None,
                     quality_passed=job['quality_passed'], error=job.get('error'),
                     traceback=job.get('traceback'))
        state['pending_keyframes'] = [k['frame'] for k in attempt.get('keyframes', []) if k['review'] is None]
        if job.get('best_attempt') is not None:
            state['outputs'] = self.output_paths(job)
        return state

    def job_result(self, job_id):
        with self.lock:
            job = copy.deepcopy(self.get(job_id))
        outputs = self.output_paths(job)
        files = [outputs[key] for key in ('job_json','validation_video','analysis')
                 if outputs.get(key) and (self.root/outputs[key]).is_file()]
        files += [relative for relative in outputs.get('previews',{}).values()
                  if (self.root/relative).is_file()]
        archive = (Path(job['directory'])/'result.tar.gz').relative_to(self.root).as_posix()
        if (self.root/archive).is_file():
            files.append(archive)
        rgba_count = None
        best = job.get('best_attempt')
        if best is not None and best < len(job['attempts']) and 'rgba_dir' in job['attempts'][best]:
            rgba_count = len(list(Path(job['attempts'][best]['rgba_dir']).glob('*.png')))
        return dict(job_id=job_id, status=self.public_status(job['status']),
                    internal_status=job['status'], quality_passed=job['quality_passed'],
                    error=job.get('error'), outputs=outputs, files=files,
                    rgba_file_count=rgba_count)

    def newest_mtime(self, directory):
        newest = 0.0
        for path in Path(directory).rglob('*'):
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
        return newest

    def package(self, job):
        """Build <job dir>/result.tar.gz (job.json + best attempt as result/).

        Reuses the archive when every packaged artifact is unchanged; the partial file
        is dot-prefixed so /download can never serve an unfinished archive.
        """
        folder = Path(job['directory'])
        archive = folder/'result.tar.gz'
        best_dir = Path(job['attempts'][job['best_attempt']]['directory'])
        manifest = folder/'job.json'
        newest = max(self.newest_mtime(best_dir), manifest.stat().st_mtime if manifest.is_file() else 0.0)
        if archive.is_file() and archive.stat().st_mtime >= newest:
            return archive
        temporary = folder/f'.result.tar.gz.{uuid.uuid4().hex}.partial'
        try:
            import io
            with tarfile.open(temporary,'w:gz') as tar:
                data = json.dumps(job,indent=2,ensure_ascii=False).encode()
                info = tarfile.TarInfo('job.json')
                info.size = len(data)
                tar.addfile(info,io.BytesIO(data))
                tar.add(best_dir,arcname='result')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
        return archive

    def submit(self, video_path, text_prompt, output_dir=None, top_k=8, random_check_frames=4,
               max_retry=2, prompt_frames=None, candidate_rank=0, score_threshold=.0,
               max_memories=6, max_pointers=15, object_score_threshold=.0,
               input_location='ubuntu', output_location='ubuntu'):
        if input_location != 'ubuntu':
            raise ValueError('Mac videos must live on the server: POST the file to /upload (or place it '
                             'under a --input-root), then pass the returned relative path with input_location=ubuntu')
        if output_location not in ('ubuntu','mac'):
            raise ValueError('output_location must be ubuntu or mac')
        video = self.resolve_video(video_path)
        if not text_prompt.strip() or not 0 <= max_retry <= 10:
            raise ValueError('Nonempty text_prompt and max_retry in 0..10 required')
        config = dict(text_prompt=text_prompt, prompt_frames=sorted(set(prompt_frames or [0])),
                      candidate_rank=candidate_rank, score_threshold=score_threshold,
                      max_memories=max_memories, max_pointers=max_pointers,
                      object_score_threshold=object_score_threshold)
        self.validate(config, 0)
        if not 1 <= top_k <= 100 or not 0 <= random_check_frames <= 100:
            raise ValueError('top_k must be 1..100; random_check_frames must be 0..100')
        job_id = uuid.uuid4().hex
        folder = (self.root/output_dir).resolve() if output_dir else self.root/job_id
        if not folder.is_relative_to(self.root) or folder == self.root:
            raise ValueError('output_dir must be a new directory inside --work-root')
        with self.lock:
            folder.mkdir(parents=True, exist_ok=False)
            job = dict(job_id=job_id, directory=str(folder), video_path=str(video),
                       output_location=output_location, max_retry=max_retry, top_k=top_k,
                       random_check_frames=random_check_frames, status='queued', attempts=[],
                       quality_passed=False, best_attempt=None, retry_history=[])
            self.jobs[job_id] = job
            self.launch(job, config, None, 'initial segmentation')
        return self.result(job_id)

    @staticmethod
    def validate(config, start):
        if not config['text_prompt'].strip():
            raise ValueError('text_prompt cannot be empty')
        if (start not in config['prompt_frames'] or len(config['prompt_frames']) > 32
                or any(type(i) is not int or i < 0 for i in config['prompt_frames'])):
            raise ValueError('Use at most 32 nonnegative integer prompt frames, including the range start')
        if not 0 <= config['candidate_rank'] <= 3 or not 0 <= config['score_threshold'] <= 1:
            raise ValueError('candidate_rank must be 0..3; score_threshold must be 0..1')
        if not 1 <= config['max_memories'] <= 32 or not 1 <= config['max_pointers'] <= 64:
            raise ValueError('Invalid memory/pointer history limits')

    def launch(self, job, config, frame_range, reason):
        index = len(job['attempts'])
        folder = Path(job['directory'])/f'attempt_{index:02d}'
        folder.mkdir()
        (folder/'rgba').mkdir()
        parent = job['best_attempt'] if frame_range else None
        attempt = dict(index=index, directory=str(folder), config=config, frame_range=frame_range,
                       parent_attempt=parent, reason=reason, status='queued', visual_review=None, keyframes=[])
        job['attempts'].append(attempt)
        job.update(status='queued', quality_passed=False)
        job.pop('error', None)
        job.pop('traceback', None)
        self.save(job)
        self.executor.submit(self.run, job['job_id'], index)

    def run(self, job_id, index):
        with self.lock:
            job = self.get(job_id)
            attempt = job['attempts'][index]
            job['status'] = attempt['status'] = 'running'
            attempt.update(processed_frames=0, total_frames=video_frame_count(job['video_path']))
            self.save(job)
        directory = Path(attempt['directory'])
        LOG.info('Starting job=%s attempt=%s',job_id,index)
        counter = dict(count=0, time=0.0)
        def progress(count):
            # Persist at most every ~2s / 64 frames so Lustre writes stay cheap.
            with self.lock:
                attempt['processed_frames'] = count
                now = time.monotonic()
                if count-counter['count'] >= 64 or now-counter['time'] >= 2:
                    counter.update(count=count, time=now)
                    self.save(job)
        try:
            if not attempt['keyframes']:
                keyframes = self.backend.prepare_keyframes(job['video_path'], directory/'keyframes', attempt['config'])
                if (len(keyframes) != len(attempt['config']['prompt_frames']) or
                        {k['frame'] for k in keyframes} != set(attempt['config']['prompt_frames'])):
                    raise ValueError('Missing keyframe candidates')
                with self.lock:
                    attempt.update(keyframes=keyframes, status='awaiting_keyframe_review')
                    job['status'] = 'awaiting_keyframe_review'
                    self.save(job)
                return
            if not all(k['review'] and k['review']['passed'] for k in attempt['keyframes']):
                raise ValueError('Every keyframe must pass visual review before tracking')
            parent = None
            if attempt['parent_attempt'] is not None:
                parent = Path(job['attempts'][attempt['parent_attempt']]['directory'])/'rgba'
            actual = self.backend.predict(job['video_path'], directory/'rgba', attempt['config'],
                                          parent, attempt['frame_range'], progress,
                                          approved_keyframes={k['frame']: directory/'keyframes'/f"{k['frame']:08d}.npy"
                                                              for k in attempt['keyframes']})
            seams = []
            if attempt['frame_range']:
                seams = [i+d for i in attempt['frame_range'] for d in (-1,0,1)]
            report = analyze(job['video_path'], directory, job['top_k'], job['random_check_frames'], [*seams, *actual])
            with self.lock:
                attempt.update(status='awaiting_visual_review', actual_prompt_frames=actual,
                    rgba_dir=str(directory/'rgba'), overlay_path=str(directory/'overlay.mp4'),
                    analysis_path=str(directory/'analysis.json'),
                    preview_paths=[str(directory/'previews'/f'{i:08d}.png') for i in report['preview_frames']],
                    preview_frames=report['preview_frames'], viewed_frames=[],
                    anomaly_frames=report['anomaly_frames'], numerical_passed=report['numerical_passed'],
                    mean_anomaly_score=report['mean_anomaly_score'], frame_count=report['frame_count'],
                    total_frames=report['frame_count'], processed_frames=report['frame_count'])
                self.select_best(job)
                job['status'] = 'awaiting_visual_review'
                self.save(job)
        except Exception as exc:
            LOG.exception('Job %s attempt %s failed',job_id,index)
            # Incomplete artifacts are not candidates for best-result selection.
            shutil.rmtree(directory/'rgba',ignore_errors=True)
            with self.lock:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                               traceback=traceback.format_exc())
                job.update(status='failed', error=attempt['error'], traceback=attempt['traceback'])
                self.save(job)

    def keyframe_preview(self, job_id, attempt_index, frame):
        with self.lock:
            job = self.get(job_id)
            attempt, keyframe = self.pending_keyframe(job, attempt_index, frame)
            data = (Path(attempt['directory'])/'keyframes'/f'{frame:08d}.png').read_bytes()
            keyframe['viewed'] = True
            self.save(job)
            return data

    @staticmethod
    def pending_keyframe(job, attempt_index, frame):
        if (job['status'] != 'awaiting_keyframe_review' or
                attempt_index != len(job['attempts'])-1):
            raise ValueError('This attempt is not awaiting keyframe review')
        attempt = job['attempts'][attempt_index]
        keyframe = next((k for k in attempt['keyframes'] if k['frame'] == frame), None)
        if keyframe is None or keyframe['review'] is not None:
            raise ValueError('Frame is not a pending keyframe')
        return attempt, keyframe

    def review_keyframe(self, job_id, attempt_index, frame, passed, notes):
        with self.lock:
            job = self.get(job_id)
            attempt, keyframe = self.pending_keyframe(job, attempt_index, frame)
            if not keyframe['viewed']:
                raise ValueError('Retrieve get_keyframe_preview before submitting review')
            if type(passed) is not bool or not notes.strip():
                raise ValueError('Supply a boolean decision and visual review notes')
            keyframe['review'] = dict(passed=passed, notes=notes)
            if not passed:
                attempt['status'] = 'keyframe_rejected'
                job['status'] = ('retry_exhausted' if len(job['attempts'])-1 >= job['max_retry']
                                 else 'needs_retry')
            elif all(k['review'] and k['review']['passed'] for k in attempt['keyframes']):
                job['status'] = attempt['status'] = 'queued'
                self.save(job)
                self.executor.submit(self.run, job_id, attempt_index)
            self.save(job)
        return self.result(job_id)

    def select_best(self, job):
        completed = [a for a in job['attempts'] if 'mean_anomaly_score' in a]
        if not completed:
            return
        # Visual acceptance takes precedence; otherwise fewer reported problems,
        # then numerical anomaly mean. Unreviewed results are never marked passed.
        def rank(a):
            review = a['visual_review']
            return (0 if review and review['passed'] else 1 if review is None else 2,
                    len(review['issues']) if review else 0, a['mean_anomaly_score'])
        best = min(completed, key=rank)
        job.update(best_attempt=best['index'], final_text_prompt=best['config']['text_prompt'],
                   final_tracking=best['config'], rgba_dir=best['rgba_dir'],
                   overlay_path=best['overlay_path'], preview_paths=best['preview_paths'],
                   analysis_path=best['analysis_path'], anomaly_frames=best['anomaly_frames'],
                   remaining_issues=(best['visual_review'] or {}).get('issues',[]))

    def review(self, job_id, attempt_index, passed, inspected_frames, issues, notes):
        with self.lock:
            job = self.get(job_id)
            if job['status'] in ('queued','running','awaiting_keyframe_review'):
                raise ValueError('Wait for the active attempt to finish')
            attempt = job['attempts'][attempt_index]
            if attempt['status'] != 'awaiting_visual_review':
                raise ValueError('This attempt is not awaiting visual review')
            if not set(attempt['preview_frames']).issubset(inspected_frames):
                raise ValueError('Inspect all generated previews before submitting review')
            if not set(attempt['preview_frames']).issubset(attempt['viewed_frames']):
                raise ValueError('Retrieve each preview using get_preview first')
            if not notes.strip() or (passed and issues) or (not passed and not issues):
                raise ValueError('Supply review notes; rejected reviews require issues; passed reviews require no issues')
            for issue in issues:
                if (type(issue.get('frame')) is not int or not 0 <= issue['frame'] < attempt['frame_count']
                        or not isinstance(issue.get('reason'),str) or not issue['reason'].strip()):
                    raise ValueError('Each issue requires a valid frame and nonempty reason')
            attempt['visual_review'] = dict(passed=passed, inspected_frames=inspected_frames, issues=issues, notes=notes)
            attempt['status'] = 'accepted' if passed else 'rejected'
            self.select_best(job)
            job['quality_passed'] = bool(job['attempts'][job['best_attempt']]['visual_review'] and
                                          job['attempts'][job['best_attempt']]['visual_review']['passed'])
            job['status'] = 'complete' if job['quality_passed'] else (
                'retry_exhausted' if len(job['attempts'])-1 >= job['max_retry'] else 'needs_retry')
            self.save(job)
        return self.result(job_id)

    def retry(self, job_id, reason, text_prompt=None, prompt_frames=None, frame_range=None,
              candidate_rank=None):
        with self.lock:
            job = self.get(job_id)
            if job['status'] not in ('needs_retry','failed'):
                raise ValueError('Retry requires rejected visual review or a failed attempt')
            if len(job['attempts'])-1 >= job['max_retry']:
                raise ValueError('max_retry reached; current best result is retained')
            if not reason.strip():
                raise ValueError('A retry reason is required')
            config = copy.deepcopy(job['attempts'][-1]['config'])
            if text_prompt is not None:
                config['text_prompt'] = text_prompt
            if candidate_rank is not None:
                config['candidate_rank'] = candidate_rank
            start = 0
            if frame_range is not None:
                if job['best_attempt'] is None:
                    raise ValueError('Partial retry needs a completed result')
                total = job['attempts'][job['best_attempt']]['frame_count']
                if (len(frame_range) != 2 or any(type(i) is not int for i in frame_range)
                        or not 0 <= frame_range[0] <= frame_range[1] < total):
                    raise ValueError('frame_range must be inclusive [start,end] inside the video')
                start = frame_range[0]
                config['prompt_frames'] = [i for i in config['prompt_frames'] if start <= i <= frame_range[1]]
            if prompt_frames is not None:
                config['prompt_frames'] = sorted(set(prompt_frames))
            if start not in config['prompt_frames']:
                config['prompt_frames'] = sorted([start,*config['prompt_frames']])
            if frame_range and any(i < start or i > frame_range[1] for i in config['prompt_frames']):
                raise ValueError('Partial retry prompt frames must be inside frame_range')
            self.validate(config,start)
            job['retry_history'].append(dict(attempt=len(job['attempts']), reason=reason,
                                             before=job['attempts'][-1]['config'], after=config,
                                             frame_range=frame_range))
            self.launch(job,config,frame_range,reason)
        return self.result(job_id)


def build_server(args):
    from mcp.server.fastmcp import FastMCP, Image
    from mcp.server.transport_security import TransportSecuritySettings
    service = None

    @asynccontextmanager
    async def lifespan(_):
        nonlocal service
        # Both load and inference use the one dedicated GPU worker.
        service = Service(None,args.work_root,args.input_root)
        try:
            service.backend = service.executor.submit(SAM3Backend,args.weights,args.device,
                                                       args.base_size,args.float32).result()
            yield {}
        finally:
            service.executor.shutdown(wait=True)

    # Accept any HTTP Host/Origin; network access is controlled outside MCP.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp = FastMCP('SAM3 video segmentation',host=args.host,port=args.port,lifespan=lifespan,
                  transport_security=security)

    @mcp.tool()
    def segment_video(video_path: str, text_prompt: str, output_dir: str | None = None,
                      top_k: int = 8, random_check_frames: int = 4, max_retry: int = 2,
                      prompt_frames: list[int] | None = None, candidate_rank: int = 0,
                      score_threshold: float = 0.0, max_memories: int = 6, max_pointers: int = 15,
                      object_score_threshold: float = 0.0, input_location: str = 'ubuntu',
                      output_location: str = 'ubuntu') -> dict:
        """Submit SAM3 -> RGBA -> temporal QA -> overlay/previews; returns job_id
        immediately. Inference runs in the background on one serialized GPU queue
        (multiple jobs may be submitted, but only one SAM3 inference runs at a time).
        Poll get_job_status for status and frame progress, then get_segmentation for
        artifacts and review state once completed.

        Frame indices are zero-based. The first prompt frame must be 0. Candidate rank
        selects among the top four nonempty detections sorted by confidence. video_path
        accepts the relative path returned by POST /upload (e.g. "abc123/video.mp4") or
        an absolute path inside a configured --input-root; never base64-encode videos or
        transfer them via SSH/SCP/SFTP. output_dir is always Ubuntu staging storage.
        output_location=mac requests an archive for subsequent download, not a Mac path.
        First inspect EVERY keyframe using get_keyframe_preview and submit_keyframe_review.
        Tracking waits until all keyframes pass; rejected candidates require rerun_segmentation.
        Then continue by viewing ALL previews and submitting visual review; never infer visual
        correctness from numerical_passed. Use rerun_segmentation after a rejected review.
        """
        parameters = locals().copy()
        parameters.pop("service", None)
        return service.submit(**parameters)

    @mcp.tool()
    def get_segmentation(job_id: str) -> dict:
        """Poll background task; returns best artifacts, anomalies, reviews and retry history."""
        return service.result(job_id)

    @mcp.tool()
    def get_job_status(job_id: str) -> dict:
        """Poll async job progress: status queued/running/completed/failed,
        internal_status, processed_frames/total_frames/progress (0..1, best effort:
        total from container metadata, exact frame count at completion). Failed jobs
        carry error plus full traceback. Completed jobs also carry outputs; use
        get_job_result to list downloadable artifact paths. internal_status=awaiting_keyframe_review
        (public status=running) requires reviewing pending_keyframes to resume tracking."""
        return service.job_status(job_id)

    @mcp.tool()
    def get_job_result(job_id: str) -> dict:
        """List job artifacts: outputs (rgba_dir, validation_video, analysis, previews,
        job.json), the existing files, and rgba_file_count. Every path is
        work-root-relative for GET /download/<path>; package_result bundles them."""
        return service.job_result(job_id)

    @mcp.tool()
    def get_keyframe_preview(job_id: str, attempt_index: int, frame: int) -> Image:
        """Inspect the exact candidate BEFORE it enters tracking memory: original left, overlay right.
        Check target identity, coverage, boundaries and unwanted foreground/background inclusion.
        Every prompt frame must be inspected, including frame zero and partial retry start.
        """
        return Image(data=service.keyframe_preview(job_id, attempt_index, frame), format='png')

    @mcp.tool()
    def submit_keyframe_review(job_id: str, attempt_index: int, frame: int, passed: bool, notes: str) -> dict:
        """Record visual correctness after get_keyframe_preview; numerical confidence is insufficient.
        All keyframes must pass before tracking starts with these exact masks. Rejecting any
        candidate stops this attempt; call rerun_segmentation with corrected text/rank/frames.
        Explain the visual decision in notes. Final video visual review is still required.
        """
        return service.review_keyframe(job_id, attempt_index, frame, passed, notes)

    @mcp.tool()
    def get_preview(job_id: str, attempt_index: int, frame: int) -> Image:
        """Return an actual MCP image for Codex visual inspection, not an Ubuntu-only path."""
        with service.lock:
            job = service.get(job_id)
            attempt = job['attempts'][attempt_index]
            if frame not in attempt.get('preview_frames',[]):
                raise ValueError('Frame is not a generated preview')
            data = (Path(attempt['directory'])/'previews'/f'{frame:08d}.png').read_bytes()
            attempt['viewed_frames'] = sorted(set(attempt['viewed_frames']) | {frame})
            service.save(job)
        return Image(data=data,format='png')

    @mcp.tool()
    def submit_visual_review(job_id: str, attempt_index: int, passed: bool,
                             inspected_frames: list[int], issues: list[dict], notes: str) -> dict:
        """Record Codex's visual decision AFTER viewing all previews. Issues: [{frame: 42,
        reason: 'mask includes hair'}]. Explain numerical false positives in notes if passing.
        Preview inspection is sampled, not a guarantee that every frame is correct.
        """
        return service.review(job_id,attempt_index,passed,inspected_frames,issues,notes)

    @mcp.tool()
    def rerun_segmentation(job_id: str, reason: str, text_prompt: str | None = None,
                           prompt_frames: list[int] | None = None, frame_range: list[int] | None = None,
                           candidate_rank: int | None = None) -> dict:
        """Bounded retry after keyframe/video visual rejection or failure. Change text, candidate rank or key frames.
        Inclusive frame_range optionally reruns only that interval and copies all other
        RGBA frames from best attempt. Tracking is reinitialized at range start; inspect
        temporal seams in the new full-video QA. max_retry is enforced across all attempts.
        """
        return service.retry(job_id,reason,text_prompt,prompt_frames,frame_range,candidate_rank)

    @mcp.tool()
    def package_result(job_id: str) -> dict:
        """Package current best RGBA, MP4, analysis, previews and full history for HTTP download.
        Reuses the existing result.tar.gz while its artifacts are unchanged. Can export
        rejected results after retry exhaustion; check quality_passed in response.
        download is work-root-relative: fetch it with GET /download/<download>.
        """
        with service.lock:
            job = copy.deepcopy(service.get(job_id))
            if job['status'] in ('queued','running','awaiting_keyframe_review') or job['best_attempt'] is None:
                raise ValueError('Wait for a completed attempt')
        archive = service.package(job)
        relative, url = download_reference(service.root, archive)
        return dict(archive_path=str(archive), download=relative, download_url=url,
                    quality_passed=job['quality_passed'], output_location=job['output_location'],
                    transfer_status='ready_for_http_download')

    # Data plane beside the MCP control plane: /upload -> input-root,
    # /download and /files/info -> work-root (see sam3_http_files.py).
    register_file_routes(mcp,args.input_root[0],args.work_root,args.max_upload_bytes)

    return mcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',default='model_weights/sam3.pt')
    parser.add_argument('--host',default='127.0.0.1',
                        help='Bind address: loopback by default; use a Tailscale/LAN IP for direct access')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--work-root',default='./mcp_jobs')
    parser.add_argument('--input-root',action='append',required=True)
    parser.add_argument('--max-upload-bytes',type=int,default=DEFAULT_MAX_UPLOAD_BYTES,
                        help='Largest accepted upload in bytes (default 64 GiB)')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--base-size',type=int,default=2048)
    parser.add_argument('--float32',action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    build_server(args).run(transport='streamable-http')


if __name__ == '__main__':
    main()
