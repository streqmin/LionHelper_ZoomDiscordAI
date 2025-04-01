import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
import webvtt
import requests
from datetime import datetime
from dotenv import load_dotenv
import traceback
import time
import re
from io import StringIO, BytesIO
import sys
from app.tasks import analyze_vtt_task, client
from anthropic import Anthropic

# 환경 변수 로드
load_dotenv()

# Anthropic 클라이언트 초기화
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

sys.setrecursionlimit(10000)  # 재귀 깊이 제한 증가

class AnthropicAPI:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
            "content-type": "application/json"
        }

    def create_completion(self, prompt, model="claude-3-haiku-20240307", max_tokens=4096, temperature=0.7):
        try:
            # 프롬프트에서 Human/Assistant 부분 추출
            parts = prompt.split("\n\nHuman: ")
            if len(parts) > 1:
                system_prompt = parts[0].strip()
                user_message = parts[1].split("\n\nAssistant:")[0].strip()
            else:
                system_prompt = ""
                user_message = prompt.strip()

            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": user_message
            })

            response = requests.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json={
                    "messages": messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            # HTTP 오류 체크
            response.raise_for_status()
            
            # 응답 파싱
            result = response.json()
            if "content" not in result or len(result["content"]) == 0:
                print(f"예상치 못한 API 응답 형식: {result}")
                return None
                
            return result["content"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 중 오류 발생: {str(e)}")
            if response := getattr(e, 'response', None):
                print(f"응답 상태 코드: {response.status_code}")
                print(f"응답 내용: {response.text}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {str(e)}")
            return None
        except Exception as e:
            print(f"예상치 못한 오류 발생: {str(e)}")
            return None

# Anthropic API 클라이언트 초기화
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
client = AnthropicAPI(anthropic_api_key)

def split_text(text, max_chunk_size=8000):
    """텍스트를 더 큰 청크로 나눕니다."""
    # 문장 단위로 분할
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def analyze_text_chunk(chunk):
    try:
        system_prompt = """다음 강의 내용을 분석하여 아래 형식으로 정리해주세요:

1. 강의 내용 요약
- 핵심 주제와 개념을 간단히 정리
- 중요한 설명이나 예시 포함

2. 강의에서 어려웠던 점
- 설명이 불충분하거나 복잡한 내용
- 이해하기 어려운 개념이나 용어

3. 강사의 발언 중 위험한 표현
- 부적절하거나 오해의 소지가 있는 표현
- 수정이 필요한 설명이나 예시

각 섹션은 bullet point(•)로 작성하고, 중요한 내용만 간단히 정리해주세요."""

        prompt = f"\n\nHuman: {system_prompt}\n\n강의 내용:\n{chunk}\n\nAssistant:"
        response = client.create_completion(prompt=prompt)
        
        if response is None:
            return "분석 결과를 가져올 수 없습니다."
            
        return response

    except Exception as e:
        print(f"텍스트 분석 중 오류 발생: {str(e)}")
        return "분석 중 오류가 발생했습니다."

def combine_analyses(analyses):
    """청크별 분석 결과를 통합합니다."""
    combined = {
        "summary": [],
        "difficulties": [],
        "risks": []
    }
    
    for analysis in analyses:
        parts = analysis.split("\n")
        current_section = None
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # 정확한 섹션 제목 매칭
            if "1. 강의 내용 요약" in part:
                current_section = "summary"
            elif "2. 강의에서 어려웠던 점" in part:
                current_section = "difficulties"
            elif "3. 강사의 발언 중 위험한 표현" in part:
                current_section = "risks"
            elif part.startswith("•") or part.startswith("-") or part.startswith("*"):
                if current_section:
                    # 중복 제거를 위해 기존 항목과 비교
                    cleaned_part = part.replace("•", "").replace("-", "").replace("*", "").strip()
                    # 이미 존재하는 항목인지 확인 (내용 기반)
                    exists = any(
                        cleaned_part in existing.replace("•", "").replace("-", "").replace("*", "").strip()
                        for existing in combined[current_section]
                    )
                    if not exists:
                        combined[current_section].append(f"• {cleaned_part}")
    
    # 결과 조합
    result = "1. 강의 내용 요약\n"
    if combined["summary"]:
        result += "\n".join(combined["summary"]) + "\n"
    else:
        result += "• 강의 내용이 추출되지 않았습니다.\n"
    result += "\n"
    
    result += "2. 강의에서 어려웠던 점\n"
    if combined["difficulties"]:
        result += "\n".join(combined["difficulties"]) + "\n"
    else:
        result += "• 특별히 어려운 점이 발견되지 않았습니다.\n"
    result += "\n"
    
    result += "3. 강사의 발언 중 위험한 표현\n"
    if combined["risks"]:
        result += "\n".join(combined["risks"]) + "\n"
    else:
        result += "• 위험한 표현이 발견되지 않았습니다.\n"
    
    return result

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

def process_curriculum_file(curriculum_file):
    """커리큘럼 파일을 처리하여 JSON 형식으로 변환합니다."""
    try:
        if curriculum_file.filename.endswith('.json'):
            curriculum_content = curriculum_file.read().decode('utf-8')
            return curriculum_content
        else:
            df = pd.read_excel(curriculum_file)
            print("엑셀 파일 읽기 성공")
            print(f"컬럼: {df.columns.tolist()}")
            print(f"데이터 행 수: {len(df)}")
            
            curriculum_data = {
                "units": []
            }
            
            for _, row in df.iterrows():
                subject_name = row['교과목명']
                details = row['세부내용']
                print(f"처리된 교과목: {subject_name}")
                
                curriculum_data["units"].append({
                    "subject_name": subject_name,
                    "details": details
                })
            
            print("JSON 변환 결과:")
            print(json.dumps(curriculum_data, ensure_ascii=False, indent=2))
            return json.dumps(curriculum_data, ensure_ascii=False)
            
    except Exception as e:
        print(f"커리큘럼 파일 처리 중 오류 발생: {str(e)}")
        raise ValueError("커리큘럼 파일 처리에 실패했습니다.")

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
            summary_response = client.create_completion(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                prompt=f"\n\nHuman: {summary_prompt}\n\nAssistant:"
            )
            
            if summary_response is None:
                summary = "강의 내용 분석 결과를 생성할 수 없습니다."
            else:
                summary = summary_response.strip()
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

def analyze_curriculum(curriculum_content, vtt_content):
    try:
        if not curriculum_content.strip():
            raise ValueError("커리큘럼 파일이 비어있습니다.")
        
        # 커리큘럼에서 교과목 정보 추출
        topics = extract_curriculum_topics(curriculum_content)
        if not topics:
            raise ValueError("커리큘럼 분석에 실패했습니다.")
        
        # 교과목 매칭 분석
        subject_matches, details_matches, summary = analyze_curriculum_match(vtt_content, topics)
        
        # 매칭된 교과목과 달성도 계산
        matched_subjects = []
        for subject, match_info in details_matches.items():
            achievement_rate = (match_info['achieved'] / match_info['total']) * 100
            matched_subjects.append({
                'name': subject,
                'achievement_rate': round(achievement_rate, 2)
            })
        
        # 달성도 기준으로 정렬
        matched_subjects.sort(key=lambda x: x['achievement_rate'], reverse=True)
        
        # 결과 통합
        result = {
            "matched_subjects": matched_subjects,
            "subject_matches": subject_matches,
            "details_matches": details_matches,
            "summary": summary
        }
        
        return result
        
    except Exception as e:
        print(f"커리큘럼 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise

def excel_to_json(excel_content):
    """엑셀 파일을 JSON 형식으로 변환"""
    try:
        # BytesIO 객체로 엑셀 데이터 읽기
        excel_file = BytesIO(excel_content)
        
        # 엑셀 파일 읽기
        df = pd.read_excel(excel_file)
        print("엑셀 파일 읽기 성공")
        print(f"컬럼: {df.columns.tolist()}")
        print(f"데이터 행 수: {len(df)}")
        
        # NaN 값을 None으로 변환
        df = df.where(pd.notnull(df), None)
        
        # 필수 컬럼 확인
        required_columns = ['교과목명', '세부내용']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 '{col}'이(가) 엑셀 파일에 없습니다.")
        
        # 데이터 전처리
        units = []
        for idx, row in df.iterrows():
            if pd.notna(row['교과목명']) and pd.notna(row['세부내용']):
                # 세부내용이 여러 줄로 되어 있는 경우 처리
                details = str(row['세부내용']).strip()
                if isinstance(details, str):
                    # 줄바꿈을 기준으로 항목 분리 후 다시 결합
                    details = '\n'.join([item.strip() for item in details.split('\n') if item.strip()])
                
                unit = {
                    'subject_name': str(row['교과목명']).strip(),
                    'details': details
                }
                units.append(unit)
                print(f"처리된 교과목: {unit['subject_name']}")
        
        if not units:
            raise ValueError("유효한 교과목 데이터를 찾을 수 없습니다.")
        
        # 커리큘럼 형식에 맞게 JSON 구조화
        curriculum_json = {
            "units": units
        }
        
        json_str = json.dumps(curriculum_json, ensure_ascii=False, indent=2)
        print("JSON 변환 결과:")
        print(json_str)
        return json_str
        
    except Exception as e:
        print(f"엑셀 변환 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"엑셀 파일 변환 실패: {str(e)}")

def send_progress(message):
    """진행 상황을 클라이언트에 전송"""
    return json.dumps({
        "status": "progress",
        "message": message
    }) + "\n"

def send_complete(data):
    """최종 결과를 클라이언트에 전송"""
    return json.dumps({
        "status": "complete",
        **data
    }) + "\n"

def analyze_chat_log(chat_content):
    try:
        system_prompt = """당신은 교육 환경에서의 채팅 기록을 전문적으로 분석하는 전문가입니다.
        제공된 채팅 기록을 심층적으로 분석하여 아래 형식에 맞춰 보고서를 작성해주세요.

        1. 전체 분석
        1.1 채팅 기록의 전반적인 맥락과 주요 참여자 파악
        1.2 주요 논의 주제와 빈도 분석
        1.3 수강생 불만/어려움 분석
        1.4 잠재적 위험 요소 및 문제점

        2. 결론
        2.1 주요 발견사항 요약
        2.2 핵심 문제점 정리
        2.3 우선순위 개선과제 제시

        분석 시 주의사항:
        1. 객관적이고 데이터 기반의 분석 수행
        2. 문제의 원인과 영향을 구체적으로 파악
        3. 실행 가능한 개선방안 제시
        4. 시급성과 중요도를 고려한 우선순위 설정
        5. 보안 및 개인정보 보호 관점 고려
        """
        
        prompt = f"\n\nHuman: {system_prompt}\n\n다음 채팅 기록을 분석해주세요:\n\n{chat_content}\n\nAssistant:"
        response = client.create_completion(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            temperature=0.7,
            prompt=prompt
        )
        
        if response is None:
            return "채팅 분석 결과를 가져올 수 없습니다."
            
        return response
        
    except Exception as e:
        print(f"채팅 분석 중 오류 발생: {str(e)}")
        return "채팅 분석 중 오류가 발생했습니다."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vtt_analysis')
def vtt_analysis():
    return render_template('vtt_analysis.html')

@app.route('/chat_analysis')
def chat_analysis():
    return render_template('chat_analysis.html')

@app.route('/analyze_vtt', methods=['POST'])
def analyze_vtt():
    try:
        if 'vtt_file' not in request.files or 'curriculum_file' not in request.files:
            return jsonify({'error': 'VTT 파일과 커리큘럼 파일이 모두 필요합니다.'}), 400
            
        vtt_file = request.files['vtt_file']
        curriculum_file = request.files['curriculum_file']
        
        if vtt_file.filename == '' or curriculum_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        # 파일 내용 읽기
        vtt_content = vtt_file.read().decode('utf-8')
        curriculum_content = curriculum_file.read().decode('utf-8')
        
        # Celery 작업 시작
        task = analyze_vtt_task.delay(vtt_content, curriculum_content)
        
        return jsonify({
            'task_id': task.id,
            'status': 'started',
            'message': '분석이 시작되었습니다.'
        })
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    task = analyze_vtt_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': '분석이 시작되기를 기다리는 중...',
            'progress': 0
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': f'분석 중... ({task.info.get("step", "")})',
            'progress': task.info.get('progress', 0)
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'status': '분석이 완료되었습니다.',
            'progress': 100,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': f'오류 발생: {task.info.get("error", "알 수 없는 오류")}',
            'progress': 0
        }
    return jsonify(response)

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    try:
        if 'chat_file' not in request.files:
            return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400
            
        file = request.files['chat_file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        # 파일 내용을 메모리에 저장
        chat_content = file.read().decode('utf-8')
        
        if not chat_content.strip():
            return jsonify({'error': '파일이 비어있습니다.'}), 400
            
        # 채팅 내용 분석
        analysis_result = analyze_chat_log(chat_content)
        
        if not analysis_result:
            return jsonify({'error': '분석 결과를 생성할 수 없습니다.'}), 500
            
        return jsonify({'analysis': analysis_result})
        
    except UnicodeDecodeError:
        return jsonify({'error': '파일 인코딩이 올바르지 않습니다. UTF-8 형식의 파일을 업로드해주세요.'}), 400
    except Exception as e:
        print(f"채팅 분석 요청 처리 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'서버 오류가 발생했습니다: {str(e)}'}), 500

@app.errorhandler(500)
def internal_error(error):
    error_message = str(error)
    if "credit balance is too low" in error_message:
        return render_template('error.html', 
                             error="API 크레딧이 부족합니다. Anthropic 계정에서 크레딧을 충전해주세요.",
                             details="Anthropic API 사용을 위해 계정의 크레딧을 확인하고 충전해주세요.")
    return render_template('error.html', 
                         error="서버 오류가 발생했습니다.",
                         details=error_message)

if __name__ == '__main__':
    socketio.run(app, debug=True) 