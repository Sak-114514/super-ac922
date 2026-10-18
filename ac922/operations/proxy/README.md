# Selective local proxy wrapper

This is an **experimental, opt-in** example for using a sing-box local mixed
HTTP/SOCKS listener for selected commands. It does not enable TUN mode, change
system proxy settings, or route all AC922 traffic.

## Private inputs

`render_subscription.py` reads a local Base64 subscription response from
`~/.config/ac922-proxy/subscription.raw` and writes a mode-0600 sing-box JSON
configuration. Keep the subscription URL/token, raw response, node credentials,
and generated JSON out of Git. The renderer selects Guangdong-entry/US-exit
Trojan/TCP links plus explicitly labelled US Reality links; it does not
implement a global rule set or cover all subscription formats.

Review the subscription's transport, SNI, TLS certificate-validation policy,
DNS requirements, and node labels before rendering. The original provider
response's `allowInsecure` flag is mapped to sing-box's TLS setting. A true
value disables certificate validation for that node; understand that tradeoff
before using it.

## Local service and wrapper

`ac922-proxy.service` is a user unit for a locally installed sing-box binary.
`with-ac922-proxy.sh` starts it on demand, performs an HTTPS health check, and
only then runs the requested command with proxy environment variables. If the
health check fails, the requested command is not run and traffic does not fall
back to direct.

The original server subscription used in the AC922 experiment did not pass
real proxy requests even when loaded by the provider's Clash-formatted
configuration in official Mihomo. This code is published as reproducible
diagnostic material, not a claim that the subscription or proxy is currently
usable. Bring your own currently working plan and validate one node before
using the wrapper.
