#!/usr/bin/env python3
"""Apply saved SAM3 text prompts independently to every frame of each video in a directory."""

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run saved per-buffer SAM3 text prompts on every frame of every mp4 video in a directory."
    )
    parser.add_argument("--prompt_path", required=True, help="Path to the saved tracking-state file")
    parser.add_argument("--input_dir", required=True, help="Directory containing mp4 videos")
    parser.add_argument(
        "--pure_text_score_threshold",
        default=0.5,
        type=float,
        help="Minimum SAM3 text-detection score for each frame (default: 0.5)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    video_paths = sorted(input_dir.glob("*.mp4"))
    print(f"Found {len(video_paths)} videos")

    runner_path = Path(__file__).with_name("load_prompts_run_video_text_each_frame.py")
    for video_path in video_paths:
        print(f"Processing {video_path}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(runner_path),
                "--prompt_path",
                args.prompt_path,
                "--input_video",
                str(video_path),
                "--pure_text_score_threshold",
                str(args.pure_text_score_threshold),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
