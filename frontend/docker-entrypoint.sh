#!/bin/sh
set -e

if [ ! -d node_modules ]; then
  npm install
fi

npm run build
exec npm run start
