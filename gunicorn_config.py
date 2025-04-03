import os
import multiprocessing

# 워커 설정
workers = 2
worker_class = "sync"  # gevent에서 sync로 변경
worker_connections = 1000
timeout = 300
keepalive = 2

# 메모리 설정
max_requests = 1000
max_requests_jitter = 50

# 포트 설정
port = int(os.environ.get("PORT", 8000))
bind = f"0.0.0.0:{port}"

# 로깅 설정
accesslog = "-"
errorlog = "-"
loglevel = "info" 