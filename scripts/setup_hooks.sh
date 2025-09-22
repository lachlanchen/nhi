#!/usr/bin/env bash
set -euo pipefail

git config core.hooksPath .githooks
echo "Git hooks path set to .githooks"
echo "Pre-commit hook installed. It will block large/binary files and non-code/doc types."

