#!/bin/sh
# Usage: with-ac922-proxy curl -L -O https://example.com/file
set -eu
if [ "$#" -eq 0 ]; then
    echo 'usage: with-ac922-proxy COMMAND [ARG...]' >&2
    exit 2
fi
systemctl --user start ac922-proxy.service
proxy=http://127.0.0.1:7890
ready=0
for attempt in 1 2 3 4 5; do
    if ss -lnt | grep -q '127.0.0.1:7890'; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo 'AC922 proxy did not open localhost:7890.' >&2
    exit 1
fi
if ! curl --silent --show-error --fail --output /dev/null \
    --proxy "$proxy" --connect-timeout 5 --max-time 15 \
    https://www.gstatic.com/generate_204; then
    echo 'AC922 proxy health check failed; download was not started.' >&2
    exit 1
fi
export http_proxy="$proxy" https_proxy="$proxy" HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
export all_proxy=socks5h://127.0.0.1:7890 ALL_PROXY=socks5h://127.0.0.1:7890
export no_proxy=${AC922_NO_PROXY:-localhost,127.0.0.1}
export NO_PROXY="$no_proxy"
exec "$@"
