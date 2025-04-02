import os
from celery.schedules import crontab

class Config:
    # Flask 설정
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    
    # Celery 설정
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'Asia/Seoul'
    CELERY_ENABLE_UTC = True
    
    # Celery Beat 스케줄 설정
    CELERY_BEAT_SCHEDULE = {
        'cleanup-old-files': {
            'task': 'app.tasks.cleanup_old_files',
            'schedule': crontab(hour=0, minute=0)  # 매일 자정에 실행
        }
    }
    
    # 파일 업로드 설정
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'vtt', 'txt'}
    
    # Anthropic API 설정
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    # Discord Webhook 설정
    DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')
    
    # 파일 정리 설정
    MAX_AGE_HOURS = 24  # 24시간 이상 된 파일 삭제 