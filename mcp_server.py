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
import uuid

from sam3_mcp_pipeline import SAM3Backend, analyze, write_json

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
            job = json.loads(path.read_text())
            if job['status'] in ('queued','running'):
                job.update(status='failed', error='Server restarted during inference; submit a retry')
            self.jobs[job['job_id']] = job

    def save(self, job):
        write_json(Path(job['directory'])/'job.json', job)

    def get(self, job_id):
        if job_id not in self.jobs:
            raise ValueError('Unknown job_id')
        return self.jobs[job_id]

    def result(self, job_id):
        with self.lock:
            result = copy.deepcopy(self.get(job_id))
        result['next_step'] = ('Call get_preview for every preview frame, then submit_visual_review; '
                               'if rejected, call rerun_segmentation with corrected parameters.')
        return result

    def submit(self, video_path, text_prompt, output_dir=None, top_k=8, random_check_frames=4,
               max_retry=2, prompt_frames=None, candidate_rank=0, score_threshold=.0,
               max_memories=6, max_pointers=15, object_score_threshold=.0,
               input_location='ubuntu', output_location='ubuntu'):
        if input_location != 'ubuntu':
            raise ValueError('Upload Mac video with mcp_transfer.py first, then use its Ubuntu path and input_location=ubuntu')
        if output_location not in ('ubuntu','mac'):
            raise ValueError('output_location must be ubuntu or mac')
        video = Path(video_path).expanduser().resolve()
        if not any(video.is_relative_to(root) for root in self.input_roots) or not video.is_file():
            raise ValueError('Input must be an existing file under a configured --input-root')
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
                       parent_attempt=parent, reason=reason, status='queued', visual_review=None)
        job['attempts'].append(attempt)
        job.update(status='queued', quality_passed=False)
        self.save(job)
        self.executor.submit(self.run, job['job_id'], index)

    def run(self, job_id, index):
        with self.lock:
            job = self.get(job_id)
            attempt = job['attempts'][index]
            job['status'] = attempt['status'] = 'running'
            self.save(job)
        directory = Path(attempt['directory'])
        LOG.info('Starting job=%s attempt=%s',job_id,index)
        try:
            parent = None
            if attempt['parent_attempt'] is not None:
                parent = Path(job['attempts'][attempt['parent_attempt']]['directory'])/'rgba'
            actual = self.backend.predict(job['video_path'], directory/'rgba', attempt['config'],
                                          parent, attempt['frame_range'])
            seams = []
            if attempt['frame_range']:
                seams = [i+d for i in attempt['frame_range'] for d in (-1,0,1)]
            report = analyze(job['video_path'], directory, job['top_k'], job['random_check_frames'], seams)
            with self.lock:
                attempt.update(status='awaiting_visual_review', actual_prompt_frames=actual,
                    rgba_dir=str(directory/'rgba'), overlay_path=str(directory/'overlay.mp4'),
                    analysis_path=str(directory/'analysis.json'),
                    preview_paths=[str(directory/'previews'/f'{i:08d}.png') for i in report['preview_frames']],
                    preview_frames=report['preview_frames'], viewed_frames=[],
                    anomaly_frames=report['anomaly_frames'], numerical_passed=report['numerical_passed'],
                    mean_anomaly_score=report['mean_anomaly_score'], frame_count=report['frame_count'])
                self.select_best(job)
                job['status'] = 'awaiting_visual_review'
                self.save(job)
        except Exception as exc:
            LOG.exception('Job %s attempt %s failed',job_id,index)
            # Incomplete artifacts are not candidates for best-result selection.
            shutil.rmtree(directory/'rgba',ignore_errors=True)
            with self.lock:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                job.update(status='failed', error=attempt['error'])
                self.save(job)

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
            if job['status'] in ('queued','running'):
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
        """Start SAM3 -> RGBA -> temporal QA -> overlay/previews. Poll get_segmentation.

        Frame indices are zero-based. The first prompt frame must be 0. Candidate rank
        selects among the top four nonempty detections sorted by confidence. Mac inputs
        must first be uploaded via SSH/SFTP. output_dir is always Ubuntu staging storage.
        output_location=mac requests an archive for subsequent download, not a Mac path.
        Continue by viewing ALL previews and submitting visual review; never infer visual
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
        """Bounded retry after visual rejection. Change text, candidate rank or key frames.
        Inclusive frame_range optionally reruns only that interval and copies all other
        RGBA frames from best attempt. Tracking is reinitialized at range start; inspect
        temporal seams in the new full-video QA. max_retry is enforced across all attempts.
        """
        return service.retry(job_id,reason,text_prompt,prompt_frames,frame_range,candidate_rank)

    @mcp.tool()
    def package_result(job_id: str) -> dict:
        """Package current best RGBA, MP4, analysis, previews and full history for SFTP download.
        Can export rejected results after retry exhaustion; check quality_passed in response.
        """
        with service.lock:
            job = copy.deepcopy(service.get(job_id))
            if job['status'] in ('queued','running') or job['best_attempt'] is None:
                raise ValueError('Wait for a completed attempt')
        folder = Path(job['directory'])
        archive = folder/f'result_{uuid.uuid4().hex}.tar.gz'
        temporary = archive.with_suffix('.partial')
        try:
            with tarfile.open(temporary,'w:gz') as tar:
                import io
                data = json.dumps(job,indent=2,ensure_ascii=False).encode()
                info = tarfile.TarInfo('job.json')
                info.size = len(data)
                tar.addfile(info,io.BytesIO(data))
                tar.add(job['attempts'][job['best_attempt']]['directory'],arcname='result')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
        return dict(archive_path=str(archive),quality_passed=job['quality_passed'],
                    output_location=job['output_location'],transfer_status='ready_for_sftp_download')

    return mcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',default='model_weights/sam3.pt')
    parser.add_argument('--host',default='127.0.0.1',
                        help='Bind address: loopback by default; use a Tailscale/LAN IP for direct access')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--work-root',default='./mcp_jobs')
    parser.add_argument('--input-root',action='append',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--base-size',type=int,default=2048)
    parser.add_argument('--float32',action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    build_server(args).run(transport='streamable-http')


if __name__ == '__main__':
    main()
