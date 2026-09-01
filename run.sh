#!/usr/bin/env bash
# 快速启动脚本（Linux / macOS / Git Bash）
set -e
cd "$(dirname "$0")"
python main.py "$@"