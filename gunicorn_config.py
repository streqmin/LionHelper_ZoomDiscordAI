import os

# 워커 설정
workers = 4
worker_class = "gevent"
worker_connections = 1000
timeout = 300
keepalive = 2

# 포트 설정
port = int(os.environ.get("PORT", 8000))
bind = f"0.0.0.0:{port}" 