"""Headless adapter for this repository's SAM3, plus streaming artifact QA."""
from pathlib import Path
import json
import math
import random

import cv2
import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def write_png(path, array):
    if not cv2.imwrite(str(path), array):
        raise OSError(f'Cannot write PNG: {path}')


def frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


class SAM3Backend:
    def __init__(self, weights, device='cuda', base_size=2048, float32=False):
        import torch
        from muggled_sam.make_sam import make_sam_from_state_dict
        from muggled_sam.demo_helpers.misc import make_device_config
        self.torch = torch
        _, self.model = make_sam_from_state_dict(str(weights))
        if self.model.name != 'samv3':
            raise ValueError('Text tracking requires SAM3 weights')
        self.model.to(**make_device_config(device, float32)).eval()
        self.detector = self.model.make_detector_model().eval()
        self.encoding = dict(max_side_length=base_size, use_square_sizing=True)

    def predict(self, video, output, config, parent=None, frame_range=None):
        from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
        from muggled_sam.demo_helpers.shared_ui_layout import make_hires_mask_uint8
        import shutil
        memory = SAMVideoObjectResults.create(config['max_memories'], config['max_pointers'], 32)
        start, end = frame_range if frame_range else (0, None)
        prompt_frames = set(config['prompt_frames'])
        actual_prompts = []
        count = 0
        with self.torch.inference_mode():
            for index, frame in frames(video):
                count += 1
                target = output / f'{index:08d}.png'
                if index < start or (end is not None and index > end):
                    if parent is None:
                        raise ValueError('Partial runs require a completed parent attempt')
                    shutil.copyfile(parent / target.name, target)
                    continue
                encoded, _, _ = self.model.encode_image(frame, **self.encoding)
                if index in prompt_frames:
                    detection, _, _ = self.detector.encode_detection_image(frame, **self.encoding)
                    exemplars = self.detector.encode_exemplars(detection, text=config['text_prompt'])
                    masks, _, scores, _ = self.detector.generate_detections(detection, exemplars)
                    if masks is None or masks.shape[1] == 0:
                        raise ValueError(f'No text candidates at frame {index}; change prompt/frame')
                    masks = masks[0]
                    valid = ((masks > 0).flatten(1).sum(1) >= max(32, masks.shape[-2]*masks.shape[-1]//2000))
                    valid &= scores.flatten() >= config['score_threshold']
                    candidates = self.torch.nonzero(valid).flatten()
                    candidates = candidates[scores.flatten()[candidates].argsort(descending=True)]
                    rank = config['candidate_rank']
                    if len(candidates) <= rank:
                        raise ValueError(f'Frame {index}: no nonempty candidate at rank {rank} above threshold')
                    selected = masks[candidates[rank]]
                    memory.store_prompt_result(index, self.model.initialize_from_mask(encoded, selected))
                    memory.prevframe_buffer.clear()
                    mask = make_hires_mask_uint8(selected, frame.shape[:2])
                    actual_prompts.append(index)
                else:
                    if not memory.check_has_prompts():
                        raise ValueError('The first processed frame must be a prompt frame')
                    score, best, masks, enc, ptr = self.model.step_video_masking(encoded, **memory.to_dict())
                    mask = make_hires_mask_uint8(masks[0, int(best.item())], frame.shape[:2])
                    if float(score.item()) >= config['object_score_threshold']:
                        memory.store_frame_result(index, enc, ptr)
                    else:
                        mask[:] = 0
                bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = mask
                write_png(target, bgra)
        if not count:
            raise ValueError('Video contains no decodable frames')
        if end is not None and end >= count:
            raise ValueError('frame_range exceeds decoded video length')
        if prompt_frames - set(range(count)):
            raise ValueError('prompt_frames exceed decoded video length')
        return actual_prompts


def mask_metrics(mask, previous=None):
    mask = mask.astype(bool)
    area = int(mask.sum())
    y, x = np.where(mask)
    centroid = [float(x.mean()), float(y.mean())] if area else None
    bbox = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if area else None
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    pieces = int(sum(stats[1:, cv2.CC_STAT_AREA] >= max(4, area * .005)))
    row = dict(area=area, centroid=centroid, bbox=bbox, fragments=pieces,
               iou=1.0, area_change=0.0, centroid_jump=0.0, bbox_jump=0.0)
    reasons = []
    if area == 0:
        reasons.append('mask_missing')
    if pieces >= 8:
        reasons.append('fragmentation')
    if previous is not None:
        old, oldrow = previous
        union = int((mask | old).sum())
        row['iou'] = float((mask & old).sum() / union) if union else 1.0
        row['area_change'] = abs(area-oldrow['area']) / max(oldrow['area'], 1)
        diag = math.hypot(*mask.shape)
        if centroid and oldrow['centroid']:
            row['centroid_jump'] = math.dist(centroid, oldrow['centroid']) / diag
            row['bbox_jump'] = max(abs(a-b) for a,b in zip(bbox, oldrow['bbox'])) / diag
        if row['iou'] < .5:
            reasons.append('low_temporal_iou')
        if row['area_change'] > .5:
            reasons.append('area_change')
        if area > max(oldrow['area']*2, mask.size*.01):
            reasons.append('sudden_expansion')
        if row['centroid_jump'] > .1 or row['bbox_jump'] > .15:
            reasons.append('position_jump')
    row['reasons'] = reasons
    row['anomaly_score'] = float(max(1.0 if area == 0 else 0.0,
        1-row['iou'], min(1, row['area_change']/2), min(1, row['centroid_jump']/.2),
        min(1, row['bbox_jump']/.3), min(1, max(0, pieces-3)/10)))
    return row


def overlay(frame, mask):
    result = frame.copy()
    result[mask] = (.55*frame[mask] + .45*np.array([30,220,40])).astype('uint8')
    return result


def analyze(video, directory, top_k=8, random_check_frames=4, extra_preview_frames=()):
    """Read one frame at a time; PNG alpha is the sole mask source."""
    directory = Path(directory)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Video has invalid FPS')
    writer = None
    rows, previous = [], None
    try:
        for index, frame in frames(video):
            rgba = cv2.imread(str(directory/'rgba'/f'{index:08d}.png'), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape != (*frame.shape[:2],4):
                raise ValueError(f'Missing or invalid RGBA frame {index}')
            mask = rgba[:,:,3] > 0
            row = dict(frame=index, **mask_metrics(mask, previous))
            rows.append(row)
            previous = (mask, row)
            visual = overlay(frame, mask)
            # mp4v needs even dimensions; pad rather than crop the original frame.
            visual = cv2.copyMakeBorder(visual,0,visual.shape[0]%2,0,visual.shape[1]%2,cv2.BORDER_CONSTANT)
            if writer is None:
                writer = cv2.VideoWriter(str(directory/'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (visual.shape[1],visual.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('OpenCV cannot encode mp4v; install an FFmpeg-enabled OpenCV build')
            writer.write(visual)
    finally:
        if writer is not None:
            writer.release()
    if not rows:
        raise ValueError('No frames to analyze')
    check = cv2.VideoCapture(str(directory/'overlay.mp4'))
    try:
        if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
            raise RuntimeError('Overlay video failed frame-count validation')
    finally:
        check.release()
    highest = sorted(rows, key=lambda r: (-r['anomaly_score'],r['frame']))[:top_k]
    chosen = {r['frame'] for r in highest}
    normal = [r['frame'] for r in rows if not r['reasons'] and r['frame'] not in chosen]
    chosen.update(random.Random(0).sample(normal, min(len(normal),random_check_frames)))
    chosen.update([0,len(rows)-1])
    chosen.update(i for i in extra_preview_frames if 0 <= i < len(rows))
    (directory/'previews').mkdir()
    for index, frame in frames(video):
        if index not in chosen:
            continue
        rgba = cv2.imread(str(directory/'rgba'/f'{index:08d}.png'),cv2.IMREAD_UNCHANGED)
        panel = np.concatenate([frame,overlay(frame,rgba[:,:,3]>0)],axis=1)
        if panel.shape[1] > 1600:
            panel = cv2.resize(panel,(1600,max(1,round(panel.shape[0]*1600/panel.shape[1]))))
        cv2.putText(panel,f'frame {index} | original / overlay | score {rows[index]["anomaly_score"]:.3f}',
                    (8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        write_png(directory/'previews'/f'{index:08d}.png',panel)
    report = dict(frame_count=len(rows), frames=rows,
                  anomaly_frames=[r for r in rows if r['reasons']],
                  numerical_passed=not any(r['reasons'] for r in rows),
                  mean_anomaly_score=sum(r['anomaly_score'] for r in rows)/len(rows),
                  preview_frames=sorted(chosen))
    write_json(directory/'analysis.json',report)
    return report
