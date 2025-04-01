import os

# Redis URL 설정
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Celery 설정
broker_url = REDIS_URL
result_backend = REDIS_URL
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Asia/Seoul'
enable_utc = True

# 작업 설정
task_track_started = True
task_time_limit = 3600  # 1시간
task_soft_time_limit = 3000  # 50분 