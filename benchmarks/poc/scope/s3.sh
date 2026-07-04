#!/usr/bin/env bash
# CUSTOM, NON-GIT layer (lives in ~/gonka/experiments, not the vllm repo).
# Push/pull decode-PoC report sessions to the Supabase gonka-artifacts bucket so results
# are reproducible from anywhere. Bucket is public-read => pull needs NO secret; push uses
# SUPABASE_SECRET (~/.config/gonka/supabase.env, local 600).
#
#   s3.sh push-report <session-dir> <name>     # upload whole folder -> reports/<name>/
#   s3.sh pull-report <name> <dest-dir>        # download whole folder (public, keyless)
#   s3.sh url <name>                           # print public report.html URL
set -euo pipefail
[ -f ~/.config/gonka/supabase.env ] && source ~/.config/gonka/supabase.env
PROJECT="${SUPABASE_PROJECT_REF:-wsegqlqqkkuzrlppdbcx}"
BUCKET="${SUPABASE_BUCKET:-gonka-artifacts}"
API="https://${PROJECT}.supabase.co/storage/v1/object/${BUCKET}"
PUB="https://${PROJECT}.supabase.co/storage/v1/object/public/${BUCKET}"

ctype(){ case "$1" in *.html) echo text/html;; *.json) echo application/json;; *.md) echo text/markdown;; *.sh|*.py) echo text/plain;; *) echo application/octet-stream;; esac; }

cmd="${1:?usage: s3.sh push-report <dir> <name> | pull-report <name> <dir> | url <name>}"; shift
case "$cmd" in
  push-report)
    dir="${1:?dir}"; name="${2:?name}"
    : "${SUPABASE_SECRET:?set SUPABASE_SECRET (source ~/.config/gonka/supabase.env)}"
    ( cd "$dir" && find . -type f | sed 's|^\./||' | sort > .manifest )
    while read -r rel; do
      curl -fsS -X POST "$API/reports/$name/$rel" \
        -H "Authorization: Bearer $SUPABASE_SECRET" -H "apikey: $SUPABASE_SECRET" \
        -H "Content-Type: $(ctype "$rel")" -H "x-upsert: true" \
        --data-binary @"$dir/$rel" >/dev/null && echo "  up  $rel"
    done < "$dir/.manifest"
    echo "PUSHED -> $PUB/reports/$name/report.html" ;;
  pull-report)
    name="${1:?name}"; dest="${2:?dest}"; mkdir -p "$dest"
    curl -fsS "$PUB/reports/$name/.manifest" -o "$dest/.manifest"
    while read -r rel; do
      mkdir -p "$dest/$(dirname "$rel")"
      curl -fsS "$PUB/reports/$name/$rel" -o "$dest/$rel" && echo "  dn  $rel"
    done < "$dest/.manifest"
    echo "PULLED -> $dest (open report.html)" ;;
  url) echo "$PUB/reports/${1:?name}/report.html" ;;
  *) echo "unknown: $cmd"; exit 2 ;;
esac
