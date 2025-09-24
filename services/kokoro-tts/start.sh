# Copyright 2025 Fluently AI, Inc. DBA Gabber. All rights reserved.
# SPDX-License-Identifier: SUL-1.0

BASEDIR=$(dirname "$0")
echo "$BASEDIR"

docker stop kokoro-tts
docker rm kokoro-tts

docker build --tag kokoro-tts:latest "$BASEDIR"

docker run \
  --name kokoro-tts \
  -p 127.0.0.1:7004:7004 \
  kokoro-tts:latest
