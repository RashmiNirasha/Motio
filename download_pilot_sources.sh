#!/bin/bash

download_video () {

    ID="$1"
    HEIGHT="$2"
    FPS="$3"

    echo
    echo "======================================"
    echo "Downloading $ID : ${HEIGHT}p @ ${FPS} fps"
    echo "======================================"

    yt-dlp \
      --no-playlist \
      -f "bv*[height=${HEIGHT}][fps=${FPS}][ext=mp4]/bv*[height=${HEIGHT}][fps=${FPS}]/bv*[height=${HEIGHT}]" \
      -o "data/ChildPlay-gaze/videos/%(id)s.%(ext)s" \
      "https://www.youtube.com/watch?v=${ID}"
}

download_video "zUdMBtK2UVE" 1080 30
download_video "w6oRoyTBmU4" 720 30
download_video "LHT2zVYvObg" 1080 24
download_video "8db0Gcv3V10" 720 25
download_video "iWYy8n6mSwc" 720 25
download_video "DGls-Z4e6OI" 720 25
