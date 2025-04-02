import os

workers = 4
port = int(os.environ.get("PORT", 8000))
bind = f"0.0.0.0:{port}"
timeout = 300
worker_class = "gevent"
worker_connections = 1000
keepalive = 2 