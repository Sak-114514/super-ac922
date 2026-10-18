#!/usr/bin/env python3
"""Render a local sing-box config from a private Base64 node subscription.

Only Guangdong-entry/US-exit Trojan/TCP and explicitly labelled US Reality
nodes are included. Never print subscription URLs, server names, passwords,
or full configuration contents.
"""

import base64
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path.home() / ".config" / "ac922-proxy"
SOURCE = ROOT / "subscription.raw"
DEST = ROOT / "config.json"


def candidates():
    raw = base64.b64decode(SOURCE.read_text().strip(), validate=True).decode()
    result = []
    for line in raw.splitlines():
        try:
            url = urlsplit(line.strip())
            label = unquote(url.fragment)
            if url.scheme == "vless":
                # US direct-line Reality node: different entry host than the
                # Trojan nodes, kept as an extra urltest candidate.
                if "美国" not in label or "直连线路" not in label:
                    continue
                query = parse_qs(url.query)
                if not (url.hostname and url.port and url.username
                        and query.get("sni") and query.get("pbk")):
                    continue
                result.append({
                    "type": "vless",
                    "tag": "us-direct-%02d" % (len(result) + 1),
                    "server": url.hostname,
                    "server_port": url.port,
                    "uuid": unquote(url.username),
                    "flow": query.get("flow", [""])[0],
                    "tls": {
                        "enabled": True,
                        "server_name": query["sni"][0],
                        "reality": {
                            "enabled": True,
                            "public_key": query["pbk"][0],
                            "short_id": query.get("sid", [""])[0],
                        },
                        "utls": {"enabled": True,
                                 "fingerprint": query.get("fp", ["chrome"])[0]},
                    },
                    "domain_resolver": {"server": "local", "strategy": "ipv4_only"},
                })
                continue
            if url.scheme != "trojan" or "美国" not in label or "广东" not in label and "直连线路" not in label:
                continue
            query = parse_qs(url.query)
            if query.get("type", ["tcp"])[0] != "tcp":
                continue
            sni = query.get("sni", query.get("peer", []))[0]
            allow_insecure = query.get("allowInsecure", ["0"])[0].lower() in (
                "1", "true", "yes")
            if not (url.hostname and url.port and url.username and sni):
                continue
            result.append({
                "type": "trojan",
                "tag": "gd-us-%02d" % (len(result) + 1),
                "server": url.hostname,
                "server_port": url.port,
                "password": unquote(url.username),
                "tls": {"enabled": True, "server_name": sni,
                        "insecure": allow_insecure},
                "domain_resolver": {"server": "local", "strategy": "ipv4_only"},
            })
        except (ValueError, IndexError, KeyError):
            continue
    if not result:
        raise SystemExit("no supported Guangdong-to-US/US-direct nodes found")
    return result


def main():
    nodes = candidates()
    config = {
        "log": {"level": "warn"},
        "dns": {"servers": [{"type": "local", "tag": "local"}]},
        "inbounds": [{
            "type": "mixed", "tag": "local-proxy",
            "listen": "127.0.0.1", "listen_port": 7890,
            "set_system_proxy": False,
        }],
        "outbounds": [{
            "type": "urltest", "tag": "gd-us-auto",
            "outbounds": [node["tag"] for node in nodes],
            "url": "https://www.gstatic.com/generate_204",
            "interval": "1m", "tolerance": 100,
            "interrupt_exist_connections": False,
        }] + nodes,
        "route": {"final": "gd-us-auto"},
    }
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = DEST.with_suffix(".json.new")
    with staging.open("w") as stream:
        os.chmod(staging, 0o600)
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    staging.replace(DEST)
    print("rendered %d Guangdong-to-US candidates on localhost:7890" % len(nodes))


if __name__ == "__main__":
    main()
