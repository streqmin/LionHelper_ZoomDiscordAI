import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
import webvtt
import requests
from datetime import datetime
from dotenv import load_dotenv
import traceback
import time
import re
from io import StringIO, BytesIO

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)

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

    def create_completion(self, prompt, model="claude-instant-1", max_tokens=4096, temperature=0.7):
        try:
            response = requests.post(
                f"{self.base_url}/complete",
                headers=self.headers,
                json={
                    "prompt": prompt,
                    "model": model,
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature
                }
            )
            
            # HTTP 오류 체크
            response.raise_for_status()
            
            # 응답 파싱
            result = response.json()
            if "completion" not in result:
                print(f"예상치 못한 API 응답 형식: {result}")
                return None
                
            return result["completion"]
            
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

def extract_curriculum_topics(curriculum_content):
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
        
        system_prompt = f"""
        당신은 IT 분야의 전문가로서, 소프트웨어 개발, 디자인, 게임 개발, 모바일 앱 개발 등 다양한 기술 분야에 대한 깊은 이해를 가지고 있습니다.
        다음 커리큘럼의 교과목별 핵심 키워드를 추출해주세요.
        각 교과목의 세부내용을 바탕으로 다음과 같은 항목들을 고려하여 키워드를 추출합니다:

        1. 주요 개념과 이론
        2. 핵심 기술과 프레임워크
        3. 개발/디자인 도구와 플랫폼
        4. 프로그래밍 언어, 라이브러리, 소프트웨어
        5. 산업 표준과 방법론
        6. 디자인 패턴과 사용자 경험 원칙
        7. 최신 기술 트렌드와 실무 적용 사례

        결과는 다음과 같은 JSON 형식으로 반환해주세요:
        {{
            "subject_keywords": {{
                "교과목명1": ["키워드1", "키워드2", ...],
                "교과목명2": ["키워드1", "키워드2", ...],
                ...
            }}
        }}

        각 교과목별로 최소 3개 이상의 관련 키워드를 추출해주세요.
        키워드는 실제 강의 내용과 매칭될 수 있도록 구체적이고 현업에서 사용되는 용어로 작성해주세요.
        교과목의 성격에 따라 적절한 항목들을 선택적으로 적용하여 키워드를 추출해주세요.
        """
        
        subjects_json = {
            "subjects": [
                {"name": name, "details": details}
                for name, details in subjects.items()
            ]
        }
        
        print("Claude에 키워드 추출 요청")
        try:
            response = client.create_completion(
                model="claude-3-haiku-20240307",
                max_tokens_to_sample=4096,
                temperature=0.7,
                prompt=f"\n\nHuman: {system_prompt}\n\n{json.dumps(subjects_json, ensure_ascii=False)}\n\nAssistant:"
            )
            
            if not response or not response.strip():
                raise ValueError("API 응답이 비어있습니다.")
            
            response_text = response.strip()
            if not response_text:
                raise ValueError("API 응답 텍스트가 비어있습니다.")
            
            topics = json.loads(response_text)
            print("추출된 키워드:")
            print(json.dumps(topics, ensure_ascii=False, indent=2))
            
            if not isinstance(topics, dict) or 'subject_keywords' not in topics:
                raise ValueError("API 응답이 올바른 형식이 아닙니다.")
            
            topics['subjects_details'] = subjects
            return topics
            
        except anthropic.NotFoundError as e:
            print(f"Claude API 모델 오류: {str(e)}")
            # 기본 키워드 생성
            default_topics = {
                "subject_keywords": {
                    subject_name: ["기본 키워드"] for subject_name in subjects.keys()
                },
                "subjects_details": subjects
            }
            return default_topics
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {str(e)}")
            raise ValueError("API 응답을 JSON으로 파싱할 수 없습니다.")
            
        except Exception as e:
            print(f"키워드 추출 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            raise ValueError("키워드 추출에 실패했습니다.")
            
    except Exception as e:
        print(f"주제 추출 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return None

def analyze_curriculum_match(vtt_content, topics):
    if not topics or 'subject_keywords' not in topics:
        return {}, {}
    
    print("\n=== 커리큘럼 매칭 분석 시작 ===")
    
    subject_matches = {}
    details_matches = {}
    
    # VTT 내용 전처리
    print("\nVTT 내용 전처리 시작...")
    vtt_content = vtt_content.replace('\n', ' ').strip()
    sentences = re.split(r'(?<=[.!?요다죠음])\s+', vtt_content)
    
    # 키워드 기반 매칭
    for subject, keywords in topics['subject_keywords'].items():
        matches = []
        for keyword in keywords:
            for sentence in sentences:
                if keyword in sentence:
                    matches.append(sentence)
        if matches:
            subject_matches[subject] = len(matches)
    
    # 세부내용 매칭 - 한 번의 API 호출로 처리
    if 'subjects_details' in topics:
        all_details = []
        all_subjects = []
        current_idx = 0
        
        for subject, details in topics['subjects_details'].items():
            detail_items = [item.strip() for item in details.split('\n') if item.strip()]
            all_details.extend(detail_items)
            all_subjects.extend([subject] * len(detail_items))
        
        if all_details:
            system_prompt = f"""
            다음 강의 내용이 각 학습 세부내용을 달성했는지 판단해주세요.
            각 항목에 대해 true 또는 false로만 답변하세요.
            결과는 JSON 형식으로 반환해주세요.

            [강의 내용]
            {vtt_content}

            [학습 세부내용]
            {json.dumps(all_details, ensure_ascii=False)}

            응답 형식:
            {{
                "results": [true/false, ...]
            }}
            """
            
            try:
                response = client.create_completion(
                    model="claude-3-haiku-20240307",
                    max_tokens_to_sample=4096,
                    temperature=0.7,
                    prompt=f"\n\nHuman: {system_prompt}\n\nAssistant:"
                )
                
                if response is None:
                    raise ValueError("API 응답이 비어있습니다.")
                
                response_text = response.strip()
                if not response_text:
                    raise ValueError("API 응답이 비어있습니다.")
                
                # JSON 부분만 추출
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if not json_match:
                    raise ValueError("API 응답에서 JSON을 찾을 수 없습니다.")
                
                json_str = json_match.group(0)
                # 주석 제거
                json_str = re.sub(r'//.*?\n', '', json_str)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                
                results = json.loads(json_str)
                if not isinstance(results, dict) or 'results' not in results:
                    raise ValueError("API 응답이 올바른 형식이 아닙니다.")
                
                matches_list = results.get('results', [])
                
                # 결과를 과목별로 정리
                for subject in set(all_subjects):
                    subject_detail_count = all_subjects.count(subject)
                    subject_matches = matches_list[current_idx:current_idx + subject_detail_count]
                    subject_details = all_details[current_idx:current_idx + subject_detail_count]
                    
                    details_matches[subject] = {
                        'matches': subject_matches,
                        'total': len(subject_matches),
                        'achieved': sum(1 for m in subject_matches if m),
                        'detail_texts': subject_details
                    }
                    current_idx += subject_detail_count
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {str(e)}")
                print(f"API 응답: {response_text}")
                # JSON 파싱 실패 시 기본값 설정
                current_idx = 0
                for subject in set(all_subjects):
                    subject_detail_count = all_subjects.count(subject)
                    details_matches[subject] = {
                        'matches': [False] * subject_detail_count,
                        'total': subject_detail_count,
                        'achieved': 0,
                        'detail_texts': all_details[current_idx:current_idx + subject_detail_count]
                    }
                    current_idx += subject_detail_count
            
    # 종합 결과 생성
    summary_prompt = f"""
    다음 강의 내용과 커리큘럼 매칭 결과를 바탕으로, 강의의 주요 초점과 커버리지를 2-3문장으로 요약해주세요.
    강의가 어떤 주제에 집중되어 있고, 어떤 부분이 잘 다루어졌는지 설명해주세요.

    [강의 내용]
    {vtt_content}

    [매칭 결과]
    {json.dumps(details_matches, ensure_ascii=False)}
    """
    
    try:
        summary_response = client.create_completion(
            model="claude-3-haiku-20240307",
            max_tokens_to_sample=1000,
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
    
    return subject_matches, details_matches, summary

def analyze_vtt_content(vtt_content):
    try:
        # 강의 내용 분석
        print("강의 내용 분석 시작...")
        chunks = split_text(vtt_content)
        analysis_results = []
        
        for chunk in chunks:
            result = analyze_text_chunk(chunk)
            if result:
                analysis_results.append(result)
        
        # 분석 결과 종합
        system_prompt = """다음은 강의 내용을 분석한 결과입니다. 이를 종합하여 다음 형식으로 정리해주세요:

1. 강의 내용 요약
- 전체 강의의 핵심 주제와 개념을 간단히 정리
- 중요한 설명이나 예시 포함

2. 강의에서 어려웠던 점
- 설명이 불충분하거나 복잡한 내용
- 이해하기 어려운 개념이나 용어

3. 강사의 발언 중 위험한 표현
- 부적절하거나 오해의 소지가 있는 표현
- 수정이 필요한 설명이나 예시

각 섹션은 bullet point(•)로 작성하고, 중요한 내용만 간단히 정리해주세요."""
        
        try:
            summary_response = client.create_completion(
                prompt=f"\n\nHuman: {system_prompt}\n\n분석 결과:\n{''.join(analysis_results)}\n\nAssistant:"
            )
            
            if summary_response is None:
                summary = "강의 내용 분석 결과를 생성할 수 없습니다."
            else:
                summary = summary_response.strip()
        except Exception as e:
            print(f"종합 결과 생성 중 오류 발생: {str(e)}")
            summary = "강의 내용 분석 결과를 생성할 수 없습니다."
        
        return summary
        
    except Exception as e:
        print(f"VTT 분석 중 오류 발생: {str(e)}")
        return "VTT 분석 중 오류가 발생했습니다."

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
        response = client.create_completion(prompt=prompt)
        
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
        if 'file' not in request.files:
            return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        if not file.filename.endswith('.vtt'):
            return jsonify({'error': 'VTT 파일만 업로드 가능합니다.'}), 400
            
        # 임시 파일로 저장
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vtt"
        file.save(temp_path)
        
        try:
            # VTT 파일 읽기
            vtt = webvtt.read(temp_path)
            vtt_content = "\n".join([caption.text for caption in vtt])
            
            # VTT 내용 분석
            analysis_result = analyze_vtt_content(vtt_content)
            
            # 교과목 매칭 분석 (topics가 제공된 경우)
            topics = request.form.get('topics')
            curriculum_match = None
            if topics:
                try:
                    topics_list = json.loads(topics)
                    curriculum_match = analyze_curriculum_match(vtt_content, topics_list)
                except json.JSONDecodeError:
                    print("교과목 정보 파싱 실패")
                    curriculum_match = None
                    
            # 임시 파일 삭제
            os.remove(temp_path)
            
            return jsonify({
                'analysis': analysis_result,
                'curriculum_match': curriculum_match
            })
            
        except Exception as e:
            print(f"VTT 파일 처리 중 오류 발생: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': 'VTT 파일 처리 중 오류가 발생했습니다.'}), 500
            
    except Exception as e:
        print(f"VTT 분석 요청 처리 중 오류 발생: {str(e)}")
        return jsonify({'error': 'VTT 분석 요청 처리 중 오류가 발생했습니다.'}), 500

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        # 파일 내용 읽기
        chat_content = file.read().decode('utf-8')
        
        # 채팅 내용 분석
        analysis_result = analyze_chat_log(chat_content)
        
        return jsonify({'analysis': analysis_result})
        
    except Exception as e:
        print(f"채팅 분석 요청 처리 중 오류 발생: {str(e)}")
        return jsonify({'error': '채팅 분석 요청 처리 중 오류가 발생했습니다.'}), 500

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
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 