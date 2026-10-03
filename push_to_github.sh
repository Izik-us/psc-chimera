#!/usr/bin/env bash
# Retired: this bootstrap script could rename branches, replace remotes, and
# force-push. Use a reviewed, repository-specific release workflow instead.

set -eu

printf '%s\n' \
  "push_to_github.sh is retired and will not modify this repository or a remote." \
  "Review git status, branch, remote, and target explicitly before any push." >&2
exit 2
