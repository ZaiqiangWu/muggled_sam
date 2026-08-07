#!/usr/bin/env python3
"""Run saved per-buffer SAM3 text prompts independently on every video frame."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    target_script = Path(__file__).with_name("load_prompts_run_video.py")
    sys.argv[0] = str(target_script)
    sys.argv.append("--pure_text")
    runpy.run_path(target_script, run_name="__main__")
