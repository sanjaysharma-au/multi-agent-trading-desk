#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
dest="$HOME/.config/systemd/user"
mkdir -p "$dest"
for unit in moonshot-queue.service moonshot-queue.timer; do
  sed "s|@REPO@|$repo|g" "$repo/deploy/systemd/$unit" > "$dest/$unit"
done
systemctl --user daemon-reload
systemctl --user enable --now moonshot-queue.timer
systemctl --user list-timers moonshot-queue.timer
