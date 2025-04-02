import os
from datetime import timedelta

class Config:
    # 기본 설정
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    
    # 파일 업로드 설정
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'txt', 'vtt'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 파일 보관 기간
    MAX_AGE_HOURS = 24  # 24시간
    
    # Anthropic API 설정
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    # Discord Webhook 설정
    DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL') 