#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to prompt file",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing mp4 videos",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    video_paths = sorted(input_dir.glob("*.mp4"))

    print(f"Found {len(video_paths)} videos")

    for video_path in video_paths:
        print(f"Processing {video_path}")

        cmd = [
            "python",
            "load_prompts_run_video.py",
            "--prompt_path",
            args.prompt_path,
            "--input_video",
            str(video_path),
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()