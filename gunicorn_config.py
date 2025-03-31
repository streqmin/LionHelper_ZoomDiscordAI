import os

workers = 4
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"  # Render가 제공하는 PORT 환경 변수 사용
timeout = 300  # 5분
worker_class = "gevent" 