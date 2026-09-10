"""Headless adapter for this repository's SAM3, plus streaming artifact QA."""
from pathlib import Path
import json
import math
import random
import uuid

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


def video_frame_count(video):
    """Container-declared frame count; None when metadata is missing or unreliable."""
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    finally:
        cap.release()
    return total if total > 0 else None


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

    def prepare_keyframes(self, video, directory, config):
        """Persist exact candidate logits and previews without initializing tracking memory."""
        from muggled_sam.demo_helpers.shared_ui_layout import make_hires_mask_uint8
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pending = set(config['prompt_frames'])
        results = []
        with self.torch.inference_mode():
            for index, frame in frames(video):
                if index not in pending:
                    continue
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
                # Preview and resumed tracking use the same float32 logits, including interpolation.
                selected = masks[candidates[rank]].float()
                if not bool(self.torch.isfinite(selected).all()):
                    raise ValueError(f'Frame {index}: nonfinite candidate mask')
                mask = make_hires_mask_uint8(selected, frame.shape[:2])
                metrics = mask_metrics(mask > 0)
                if metrics['area'] == 0:
                    raise ValueError(f'Frame {index}: empty full-resolution candidate mask')
                tracking_components = self.split_tracking_masks(mask > 0, config['max_tracking_objects'])
                np.save(directory/f'{index:08d}.npy', selected.cpu().numpy(), allow_pickle=False)
                previews = []
                for candidate_rank, candidate in enumerate(candidates[:4]):
                    logits = masks[candidate].float()
                    candidate_mask = make_hires_mask_uint8(logits, frame.shape[:2])
                    previews.append(dict(candidate_rank=candidate_rank,
                                         score=float(scores.flatten()[candidate].item()), mask=candidate_mask > 0))
                preview_info = write_keyframe_previews(directory.parent, index, frame, previews, rank)
                results.append(dict(frame=index, confidence=float(scores.flatten()[candidates[rank]].item()),
                                    metrics=metrics, viewed=False, review=None,
                                    tracking_components=[dict(object_index=object_index,
                                                              area=int(component.sum()))
                                                         for object_index, component in enumerate(tracking_components)],
                                    **preview_info))
                pending.remove(index)
                if not pending:
                    break
        if pending:
            raise ValueError('prompt_frames exceed decoded video length')
        return results

    @staticmethod
    def split_tracking_masks(mask, max_objects):
        """Return the largest meaningful disconnected regions as separate object prompts."""
        mask = np.asarray(mask, dtype=bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
        minimum = max(4, round(mask.size * .001))
        components = [(int(stats[label, cv2.CC_STAT_AREA]), label)
                      for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= minimum]
        components.sort(reverse=True)
        result = [(labels == label) for _, label in components[:max_objects]]
        # The selected candidate was already nonempty. Retaining it as one object is
        # safer than producing no tracker when every piece is below the noise threshold.
        return result or [mask]

    def predict(self, video, output, config, parent=None, frame_range=None, progress=None,
                approved_keyframes=None):
        prompt_frames = set(config['prompt_frames'])
        if approved_keyframes is None or set(approved_keyframes) != prompt_frames:
            raise ValueError('Every keyframe requires visual approval before tracking')
        from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
        from muggled_sam.demo_helpers.shared_ui_layout import make_hires_mask_uint8
        import shutil
        memories = []
        start, end = frame_range if frame_range else (0, None)
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
                    if progress is not None:
                        progress(count)
                    continue
                encoded, _, _ = self.model.encode_image(frame, **self.encoding)
                if index in prompt_frames:
                    selected = self.torch.from_numpy(np.load(approved_keyframes[index], allow_pickle=False))
                    selected = selected.to(device=encoded[0].device)
                    candidate = make_hires_mask_uint8(selected, frame.shape[:2]) > 0
                    components = self.split_tracking_masks(candidate, config['max_tracking_objects'])
                    if not memories:
                        memories = [SAMVideoObjectResults.create(config['max_memories'], config['max_pointers'], 32)
                                    for _ in components]
                    # Subsequent keyframes re-prompt each extant object. If the detector
                    # joins objects during an occlusion, keep unmatched trackers intact.
                    for memory, component in zip(memories, components):
                        memory.store_prompt_result(index, self.model.initialize_from_mask(encoded, component))
                        memory.prevframe_buffer.clear()
                    mask = np.zeros(frame.shape[:2], dtype='uint8')
                    for component in components:
                        mask[component] = 255
                    actual_prompts.append(index)
                else:
                    if not memories or not all(memory.check_has_prompts() for memory in memories):
                        raise ValueError('The first processed frame must be a prompt frame')
                    mask = np.zeros(frame.shape[:2], dtype='uint8')
                    for memory in memories:
                        score, best, masks, enc, ptr = self.model.step_video_masking(encoded, **memory.to_dict())
                        component = make_hires_mask_uint8(masks[0, int(best.item())], frame.shape[:2])
                        if float(score.item()) >= config['object_score_threshold']:
                            memory.store_frame_result(index, enc, ptr)
                            mask = cv2.bitwise_or(mask, component)
                bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = mask
                write_png(target, bgra)
                if progress is not None:
                    progress(count)
        if not count:
            raise ValueError('Video contains no decodable frames')
        if end is not None and end >= count:
            raise ValueError('frame_range exceeds decoded video length')
        if prompt_frames - set(range(count)):
            raise ValueError('prompt_frames exceed decoded video length')
        return dict(prompt_frames=actual_prompts, tracking_object_count=len(memories))


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


def write_keyframe_previews(attempt_directory, frame_index, frame, candidates, selected_rank):
    """Write the up-to-four selectable candidates and a bounded comparison panel.

    Returned paths are attempt-relative; HTTP references are resolved by the service.
    """
    directory = Path(attempt_directory)/'keyframe_previews'
    directory.mkdir(parents=True, exist_ok=True)
    tiles, records = [], []
    for candidate in candidates:
        rank, score = candidate['candidate_rank'], candidate.get('score')
        tile = np.concatenate([frame, overlay(frame, candidate['mask'])], axis=1)
        if tile.shape[1] > 1920:
            tile = cv2.resize(tile, (1920, max(1, round(tile.shape[0]*1920/tile.shape[1]))))
        tile = cv2.copyMakeBorder(tile, 44, 0, 0, 0, cv2.BORDER_CONSTANT)
        score_label = f'{score:.4f}' if score is not None else 'n/a'
        label = f'Frame {frame_index} | rank {rank} | score {score_label} | original / overlay'
        if rank == selected_rank:
            label += ' | SELECTED'
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
        path = directory/f'frame_{frame_index:08d}_candidate_{rank}.png'
        write_png(path, tile)
        records.append(dict(frame=frame_index, candidate_rank=rank, score=score,
                            selected=rank == selected_rank, path=path.relative_to(attempt_directory).as_posix()))
        tiles.append(tile)
    columns = min(2, len(tiles))
    rows = (len(tiles)+columns-1)//columns
    tile_width = min(960, tiles[0].shape[1])
    tile_height = max(1, round(tiles[0].shape[0]*tile_width/tiles[0].shape[1]))
    panel = np.zeros((rows*tile_height, columns*tile_width, 3), dtype='uint8')
    for i, tile in enumerate(tiles):
        y, x = (i//columns)*tile_height, (i%columns)*tile_width
        panel[y:y+tile_height, x:x+tile_width] = cv2.resize(tile, (tile_width, tile_height))
    panel_path = directory/f'frame_{frame_index:08d}_comparison.png'
    write_png(panel_path, panel)
    return dict(candidates=records, comparison_panel=panel_path.relative_to(attempt_directory).as_posix())


def overlay(frame, mask):
    result = frame.copy()
    result[mask] = (.55*frame[mask] + .45*np.array([30,220,40])).astype('uint8')
    return result


def ensure_qa_contact_sheet(directory, preview_frames):
    """Create/cache a contact sheet from existing PNGs only; no video/model access."""
    directory = Path(directory)
    indices = sorted(set(preview_frames))
    if not indices:
        raise ValueError('No QA previews available for contact sheet')
    sources = [directory/'previews'/f'{index:08d}.png' for index in indices]
    signature = [dict(frame=index, mtime_ns=path.stat().st_mtime_ns, size=path.stat().st_size)
                 for index, path in zip(indices, sources)]
    sheet = directory/'qa_contact_sheet.png'
    cache = directory/'qa_contact_sheet.json'
    if sheet.is_file() and cache.is_file():
        try:
            if json.loads(cache.read_text()) == signature:
                return sheet
        except (OSError, ValueError):
            pass
    columns, width, header = 4, 480, 32
    first = cv2.imread(str(sources[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f'Cannot read QA preview: {sources[0].name}')
    height = min(320, max(80, round(first.shape[0]*width/first.shape[1])))
    rows = (len(indices)+columns-1)//columns
    panel = np.full((rows*(height+header), columns*width, 3), 24, dtype='uint8')
    for position, (index, source) in enumerate(zip(indices, sources)):
        preview = first if position == 0 else cv2.imread(str(source), cv2.IMREAD_COLOR)
        if preview is None:
            raise ValueError(f'Cannot read QA preview: {source.name}')
        scale = min(width/preview.shape[1], height/preview.shape[0])
        thumb = cv2.resize(preview, (max(1, round(preview.shape[1]*scale)),
                                    max(1, round(preview.shape[0]*scale))), interpolation=cv2.INTER_AREA)
        y, x = (position//columns)*(height+header), (position%columns)*width
        cv2.putText(panel, f'Frame {index}', (x+10, y+23), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 1, cv2.LINE_AA)
        top, left = y+header+(height-thumb.shape[0])//2, x+(width-thumb.shape[1])//2
        panel[top:top+thumb.shape[0], left:left+thumb.shape[1]] = thumb
    temporary = directory/f'.qa_contact_sheet.{uuid.uuid4().hex}.partial.png'
    try:
        write_png(temporary, panel)
        temporary.replace(sheet)
        write_json(cache, signature)
    finally:
        temporary.unlink(missing_ok=True)
    return sheet


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
    ensure_qa_contact_sheet(directory, chosen)
    report = dict(frame_count=len(rows), frames=rows,
                  anomaly_frames=[r for r in rows if r['reasons']],
                  numerical_passed=not any(r['reasons'] for r in rows),
                  mean_anomaly_score=sum(r['anomaly_score'] for r in rows)/len(rows),
                  preview_frames=sorted(chosen))
    write_json(directory/'analysis.json',report)
    return report
