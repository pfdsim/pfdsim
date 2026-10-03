#!/usr/bin/env bash
set -euo pipefail
if [[ ${RENEWED_DOMAINS:-} == *chemicalprocess.org* ]]; then
    nginx -t
    systemctl reload nginx
fi
