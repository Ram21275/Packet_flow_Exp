#!/usr/bin/env bash
set -euo pipefail
# Run in the repo root after gh auth login. Never store tokens in remote URLs.
if ! gh repo view Ram21275/Packet_flow_Exp >/dev/null 2>&1; then
  gh repo create Ram21275/Packet_flow_Exp --public
fi
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin https://github.com/Ram21275/Packet_flow_Exp.git
else
  git remote add origin https://github.com/Ram21275/Packet_flow_Exp.git
fi
git push -u origin main
