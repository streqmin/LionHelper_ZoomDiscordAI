from celery import Celery
from app import app
import json
import webvtt
from io import StringIO
import pandas as pd
import traceback
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# 환경 변수 로드
load_dotenv()

# Anthropic 클라이언트 초기화
client = Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    http_client=None
)

# Celery 인스턴스 생성
celery = Celery('tasks')
celery.config_from_object('app.celery_config')

# Flask 애플리케이션 컨텍스트 설정
celery.conf.update(app.config)

class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, **kwargs)

celery.Task = ContextTask

@celery.task(bind=True)
def analyze_vtt_task(self, vtt_content, curriculum_content):
    """VTT 분석 작업을 수행하는 Celery 태스크"""
    try:
        # 1. VTT 파일 분석
        analyzed_content = analyze_vtt_content(vtt_content)
        self.update_state(state='PROGRESS', meta={'step': 'vtt_analysis', 'progress': 20})
        
        # 2. 커리큘럼 주제 추출
        topics = extract_curriculum_topics(curriculum_content)
        if not topics:
            raise ValueError("커리큘럼 분석에 실패했습니다.")
        self.update_state(state='PROGRESS', meta={'step': 'curriculum_analysis', 'progress': 40})
        
        # 3. 매칭 분석
        match_analysis = analyze_curriculum_match(analyzed_content, topics)
        if not match_analysis:
            raise ValueError("매칭 분석에 실패했습니다.")
        self.update_state(state='PROGRESS', meta={'step': 'matching_analysis', 'progress': 80})
        
        # 4. 결과 생성
        result = {
            'vtt_content': analyzed_content,
            'curriculum_analysis': topics,
            'match_analysis': match_analysis
        }
        self.update_state(state='SUCCESS', meta={'step': 'complete', 'progress': 100})
        
        return result
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

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