#!/bin/bash
# scripts/setup_pod_proxy.sh - logged HTTP proxy allowlist
set -euo pipefail

apt-get install -y tinyproxy
cat > /etc/tinyproxy/tinyproxy.conf <<'EOF'
Port 8888
Listen 127.0.0.1
LogFile "/var/log/tinyproxy/proxy.log"
LogLevel Info
MaxClients 200
Allow 127.0.0.1
# Allowlist
FilterURLs Yes
Filter "/etc/tinyproxy/filter"
EOF

cat > /etc/tinyproxy/filter <<'EOF'
^https?://pypi\.org
^https?://files\.pythonhosted\.org
^https?://registry\.npmjs\.org
^https?://registry-1\.docker\.io
^https?://auth\.docker\.io
^https?://production\.cloudflare\.docker\.com
^https?://archive\.ubuntu\.com
^https?://security\.ubuntu\.com
^https?://deb\.debian\.org
^https?://security\.debian\.org
^https?://github\.com
^https?://codeload\.github\.com
^https?://objects\.githubusercontent\.com
^https?://(.*\.)?huggingface\.co
^https?://cdn-lfs.*\.huggingface\.co
EOF

systemctl restart tinyproxy
echo "proxy on 127.0.0.1:8888 with allowlist"
