#!/bin/sh

find . -name ".jupyter_cache" -print0 | xargs -0 rm -rf 2>/dev/null || true
find . -name "checkpoints" -print0 | xargs -0 rm -rf 2>/dev/null || true
find . -name "lightning_logs" -print0 | xargs -0 rm -rf 2>/dev/null || true

rm -rf _site/
rm -rf _freeze/
rm -rf site_libs/
rm -rf .quarto/
