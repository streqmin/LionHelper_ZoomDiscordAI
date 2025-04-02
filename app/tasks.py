import os
import json
from datetime import datetime, timedelta
from celery import Celery
from anthropic import Anthropic
from app.config import Config
import webvtt
from io import StringIO
import pandas as pd
import traceback
from dotenv import load_dotenv
import time
import re
from collections import Counter

# 환경 변수 로드
load_dotenv()

# Celery 인스턴스 생성
celery = Celery('app')

# 설정 직접 지정
celery.conf.update({
    'broker_url': Config.CELERY_BROKER_URL,
    'result_backend': Config.CELERY_RESULT_BACKEND,
    'task_serializer': Config.CELERY_TASK_SERIALIZER,
    'result_serializer': Config.CELERY_RESULT_SERIALIZER,
    'accept_content': Config.CELERY_ACCEPT_CONTENT,
    'timezone': Config.CELERY_TIMEZONE,
    'enable_utc': Config.CELERY_ENABLE_UTC,
    'beat_schedule': Config.CELERY_BEAT_SCHEDULE
})

# ContextTask 수정
class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

celery.Task = ContextTask

# Anthropic 클라이언트 초기화
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

@celery.task
def analyze_vtt_task(file_path):
    try:
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Claude API 호출
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.7,
            system="""당신은 줌 회의록 분석 전문가입니다. 
            주어진 VTT 파일의 내용을 분석하여 다음 형식으로 요약해주세요:
            
            # 회의 요약
            - 회의 주제:
            - 주요 참석자:
            - 회의 시간:
            - 핵심 논의 사항:
            - 결정사항:
            - 후속 조치사항:
            
            # 상세 내용
            [시간대별 주요 내용]
            
            # 주요 키워드
            [회의에서 언급된 주요 키워드들]
            
            # 액션 아이템
            [구체적인 할 일과 담당자]
            
            # 추가 참고사항
            [기타 중요한 정보나 맥락]""",
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        # 분석 결과 저장
        result = {
            'status': 'success',
            'summary': message.content[0].text,
            'timestamp': datetime.now().isoformat()
        }
        
        # 결과를 JSON 파일로 저장
        result_file = file_path.replace('.vtt', '_analysis.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

@celery.task
def cleanup_old_files():
    """오래된 파일들을 정리하는 태스크"""
    try:
        upload_dir = Config.UPLOAD_FOLDER
        max_age = timedelta(hours=Config.MAX_AGE_HOURS)
        current_time = datetime.now()
        
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_age = datetime.fromtimestamp(os.path.getmtime(file_path))
                if current_time - file_age > max_age:
                    os.remove(file_path)
                    print(f"Deleted old file: {filename}")
        
        return {'status': 'success', 'message': 'Cleanup completed'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def analyze_vtt_content(vtt_content):
    """VTT 파일 내용을 분석하여 텍스트를 추출합니다."""
    try:
        print("강의 내용 분석 시작...")
        vtt = webvtt.read_buffer(StringIO(vtt_content))
        text_content = []
        for caption in vtt:
            text_content.append(caption.text)
        return "\n".join(text_content)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        return vtt_content

def extract_curriculum_topics(curriculum_content):
    """커리큘럼에서 주제와 키워드를 추출합니다."""
    try:
        print("커리큘럼 내용 분석 시작")
        curriculum_data = json.loads(curriculum_content)
        print("JSON 파싱 성공")
        
        subjects = {}
        if isinstance(curriculum_data, dict) and 'units' in curriculum_data:
            for unit in curriculum_data['units']:
                if 'subject_name' in unit and unit['subject_name'] and 'details' in unit:
                    subjects[unit['subject_name']] = unit['details']
                    print(f"추출된 교과목: {unit['subject_name']}")
        
        if not subjects:
            raise ValueError("커리큘럼에서 교과목명과 세부내용을 찾을 수 없습니다.")
        
        # 기본 키워드 생성
        default_topics = {
            "subject_keywords": {},
            "subjects_details": subjects
        }
        
        # 각 교과목별로 기본 키워드 생성
        for subject_name, details in subjects.items():
            keywords = []
            # 세부내용을 줄바꿈으로 분리하여 각 항목을 키워드로 사용
            for detail in details.split('\n'):
                if detail.strip():
                    keywords.append(detail.strip())
            default_topics["subject_keywords"][subject_name] = keywords
        
        print("기본 키워드 생성 완료")
        print(json.dumps(default_topics, ensure_ascii=False, indent=2))
        
        return default_topics
            
    except Exception as e:
        print(f"주제 추출 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return None

def analyze_curriculum_match(vtt_content, topics):
    """VTT 내용과 커리큘럼을 매칭하여 분석합니다."""
    try:
        if not topics or 'subject_keywords' not in topics:
            raise ValueError("주제 정보가 없습니다.")
            
        subject_keywords = topics['subject_keywords']
        subjects_details = topics.get('subjects_details', {})
        
        # 각 교과목별 매칭 분석
        subject_matches = {}
        for subject_name, keywords in subject_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword.lower() in vtt_content.lower():
                    matches.append(keyword)
            
            if matches:
                subject_matches[subject_name] = {
                    'keywords': matches,
                    'details': subjects_details.get(subject_name, ''),
                    'match_count': len(matches)
                }
        
        # 매칭 결과 정렬
        sorted_matches = sorted(
            subject_matches.items(),
            key=lambda x: x[1]['match_count'],
            reverse=True
        )
        
        # 종합 분석 생성
        summary_prompt = f"""
        다음 강의 내용과 커리큘럼 매칭 결과를 바탕으로, 강의의 주요 초점과 커버리지를 2-3문장으로 요약해주세요.
        강의가 어떤 주제에 집중되어 있고, 어떤 부분이 잘 다루어졌는지 설명해주세요.

        [강의 내용]
        {vtt_content}

        [매칭 결과]
        {json.dumps(subject_matches, ensure_ascii=False)}
        """
        
        try:
            summary_response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": summary_prompt}
                ]
            )
            
            if summary_response is None:
                summary = "강의 내용 분석 결과를 생성할 수 없습니다."
            else:
                summary = summary_response.content[0].text
        except Exception as e:
            print(f"종합 결과 생성 중 오류 발생: {str(e)}")
            summary = "강의 내용 분석 결과를 생성할 수 없습니다."
        
        return {
            'subject_matches': dict(sorted_matches),
            'total_subjects': len(subject_keywords),
            'matched_subjects': len(subject_matches),
            'summary': summary
        }
        
    except Exception as e:
        print(f"매칭 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return None 

@celery.task
def analyze_chat_task(file_path):
    """채팅 로그 파일을 분석하는 태스크"""
    try:
        print(f"채팅 로그 분석 시작: {file_path}")
        
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            chat_content = f.read()
        
        # 채팅 내용 분석
        analysis_result = analyze_chat_content(chat_content)
        
        # 결과 저장
        result_file = file_path.replace('.txt', '_analysis.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        return {
            'status': 'success',
            'message': '채팅 분석이 완료되었습니다.',
            'result_file': os.path.basename(result_file),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"채팅 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

def analyze_chat_content(chat_content):
    """채팅 내용을 분석하여 통계와 인사이트를 추출합니다."""
    try:
        # 채팅 메시지 파싱
        messages = []
        for line in chat_content.split('\n'):
            if line.strip():
                # 시간과 사용자 이름 추출 (예: "[10:30] 사용자명: 메시지")
                match = re.match(r'\[(.*?)\] (.*?): (.*)', line)
                if match:
                    time_str, username, message = match.groups()
                    messages.append({
                        'time': time_str,
                        'username': username,
                        'message': message
                    })
        
        # 기본 통계
        total_messages = len(messages)
        unique_users = len(set(msg['username'] for msg in messages))
        
        # 사용자별 메시지 수
        user_message_counts = {}
        for msg in messages:
            username = msg['username']
            user_message_counts[username] = user_message_counts.get(username, 0) + 1
        
        # 시간대별 메시지 수
        time_message_counts = {}
        for msg in messages:
            hour = msg['time'].split(':')[0]
            time_message_counts[hour] = time_message_counts.get(hour, 0) + 1
        
        # 키워드 분석
        all_messages = ' '.join(msg['message'] for msg in messages)
        words = re.findall(r'\w+', all_messages.lower())
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)
        
        return {
            'summary': {
                'total_messages': total_messages,
                'unique_users': unique_users,
                'average_messages_per_user': total_messages / unique_users if unique_users > 0 else 0
            },
            'participation': {
                'user_message_counts': user_message_counts,
                'time_message_counts': time_message_counts
            },
            'keywords': {
                'common_words': common_words
            }
        }
    except Exception as e:
        print(f"채팅 내용 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return None 