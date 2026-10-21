#!/bin/bash

# Download the ChildPlay source videos listed in
# mvp/pincer_candidates/missing_videos.csv, at the
# resolution / fps the ChildPlay annotations used.
#
# Usage (from the repository root):
#   conda activate childplay_download
#   bash mvp/download_childplay_videos.sh
#
# Then re-run:
#   conda activate childplay_hand
#   python mvp/extract_pincer_candidates.py

set -u

LIST="mvp/pincer_candidates/missing_videos.csv"
OUT="data/ChildPlay-gaze/videos"

if [ ! -f "$LIST" ]; then
    echo "Missing $LIST. Run python mvp/extract_pincer_candidates.py first."
    exit 1
fi

mkdir -p "$OUT"

FAILED=()

while IFS=, read -r ID HEIGHT FPS COUNT; do

    [ "$ID" = "video_id" ] && continue

    echo
    echo "======================================"
    echo "Downloading $ID : ${HEIGHT}p @ ${FPS} fps ($COUNT candidates)"
    echo "======================================"

    # Prefer the exact annotated format, then fall back
    # to the same height. The extractor rescales crops
    # if the resolution differs and skips fps mismatches.
    if [ "$HEIGHT" -gt 0 ] 2>/dev/null; then
        FORMAT="bv*[height=${HEIGHT}][fps=${FPS}][ext=mp4]/bv*[height=${HEIGHT}][fps=${FPS}]/bv*[height=${HEIGHT}]/bv*"
    else
        FORMAT="bv*[fps=${FPS}][ext=mp4]/bv*"
    fi

    if ! yt-dlp \
        --no-playlist \
        -f "$FORMAT" \
        -o "$OUT/%(id)s.%(ext)s" \
        "https://www.youtube.com/watch?v=${ID}"; then
        FAILED+=("$ID")
    fi

done < "$LIST"

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Could not download ${#FAILED[@]} videos (removed or private): ${FAILED[*]}"
else
    echo "All videos downloaded."
fi
