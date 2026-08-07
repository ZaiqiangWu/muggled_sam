#!/usr/bin/env bash

# Remove generated segmentation artifacts while keeping source videos intact.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
videos_dir="$repo_dir/videos"
saved_frames_dir="$repo_dir/saved_images/run_video"
generated_masks_dir="$repo_dir/generated_mask_videos"

clear_directory_contents() {
    local target_dir="$1"

    if [[ -d "$target_dir" ]]; then
        find "$target_dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
        echo "Cleared: $target_dir"
    else
        echo "Skipping missing directory: $target_dir"
    fi
}

# For each ./videos/<group>/<name>.mp4, remove ./videos/<group>/<name>/.
# The source .mp4 files themselves are never removed.
if [[ -d "$videos_dir" ]]; then
    while IFS= read -r -d '' video_file; do
        frame_dir="${video_file%.mp4}"
        if [[ -d "$frame_dir" ]]; then
            rm -rf -- "$frame_dir"
            echo "Removed: $frame_dir"
        fi
    done < <(find "$videos_dir" -mindepth 2 -maxdepth 2 -type f -name '*.mp4' -print0)
else
    echo "Skipping missing directory: $videos_dir"
fi

clear_directory_contents "$saved_frames_dir"
clear_directory_contents "$generated_masks_dir"
