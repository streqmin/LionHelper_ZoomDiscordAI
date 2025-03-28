from flask import Flask, render_template, request, jsonify, Response
import openai
import webvtt
import os
import traceback
import time
import json
import pandas as pd
from dotenv import load_dotenv
from io import StringIO, BytesIO
from datetime import datetime
import re

load_dotenv()

app = Flask(__name__)

# OpenAI 클라이언트 초기화
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("경고: OpenAI API 키가 설정되지 않았습니다!")
openai.api_key = openai_api_key

def split_text(text, max_chunk_size=4000):
    """텍스트를 더 큰 청크로 나눕니다."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1
        if current_size > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def analyze_text_chunk(chunk):
    """텍스트 청크를 분석하고 재시도 로직을 포함합니다."""
    max_retries = 3
    base_delay = 10  # 대기 시간 감소
    
    system_prompt = """당신은 교육 내용 분석 전문가입니다.
    강의 내용을 분석할 때 반드시 아래 세 가지 섹션으로만 구분하여 분석해야 합니다.
    각 섹션에는 최소 1개 이상의 내용이 포함되어야 합니다.
    다른 섹션을 추가하거나 다른 형식으로 응답하지 마세요.
    
    응답 형식:
    1. 강의 내용 요약
    • 첫 번째 내용
    • 두 번째 내용
    • 세 번째 내용

    2. 강의에서 어려웠던 점
    • 첫 번째 어려움
    • 두 번째 어려움

    3. 강사의 발언 중 위험한 표현
    • 첫 번째 위험 표현
    • 두 번째 위험 표현

    """
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"다음 강의 내용을 위 형식에 맞춰 분석해주세요:\n\n{chunk}"}
                ],
                temperature=0.3
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            if "Rate limit" in str(e):
                wait_time = base_delay * (attempt + 1)
                print(f"Rate limit에 도달했습니다. {wait_time}초 대기 후 재시도합니다.")
                time.sleep(wait_time)
            else:
                raise
    
    raise Exception("최대 재시도 횟수를 초과했습니다.")

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
        당신은 데이터 분석, 통계, 머신러닝, 웹 개발 분야의 전문가입니다.
        다음 커리큘럼의 교과목별 핵심 키워드를 추출해주세요.
        각 교과목의 세부내용을 바탕으로 다음과 같은 항목들을 고려하여 키워드를 추출합니다:

        1. 주요 개념과 이론
        2. 기술적 용어와 프레임워크 (예: Vue.js, Flask, Python 등)
        3. 개발 도구와 플랫폼 (예: GitHub 등)
        4. 프로그래밍 언어와 라이브러리
        5. 웹 개발 및 AI 관련 용어

        결과는 다음과 같은 JSON 형식으로 반환해주세요:
        {{
            "subject_keywords": {{
                "교과목명1": ["키워드1", "키워드2", ...],
                "교과목명2": ["키워드1", "키워드2", ...],
                ...
            }}
        }}

        각 교과목별로 최소 3개 이상의 관련 키워드를 추출해주세요.
        키워드는 실제 강의 내용과 매칭될 수 있도록 구체적으로 작성해주세요.
        """
        
        subjects_json = {
            "subjects": [
                {"name": name, "details": details}
                for name, details in subjects.items()
            ]
        }
        
        print("GPT에 키워드 추출 요청")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(subjects_json, ensure_ascii=False)}
            ],
            temperature=0.3
        )
        
        topics = json.loads(response.choices[0].message['content'].strip())
        print("추출된 키워드:")
        print(json.dumps(topics, ensure_ascii=False, indent=2))
        
        topics['subjects_details'] = subjects
        return topics
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
    
    # 세부내용 매칭
    if 'subjects_details' in topics:
        for subject, details in topics['subjects_details'].items():
            detail_items = [item.strip() for item in details.split('\n') if item.strip()]
            matches = []
            detail_texts = []
            
            for item in detail_items:
                system_prompt = f"""
                다음 강의 내용이 주어진 학습 세부내용을 달성했는지 판단해주세요.

                [학습 세부내용]
                {item}

                [강의 내용]
                {vtt_content}

                true 또는 false로만 답변하세요.
                다른 설명은 하지 마세요.
                """
                
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "위 내용을 바탕으로 답변해주세요."}
                        ],
                        temperature=0.1
                    )
                    
                    result = response.choices[0].message['content'].strip().lower()
                    matches.append(result == 'true')
                    detail_texts.append(item)
                    
                except Exception as e:
                    print(f"세부내용 분석 중 오류 발생: {str(e)}")
                    matches.append(False)
                    detail_texts.append(item)
            
            if matches:
                details_matches[subject] = {
                    'matches': matches,
                    'total': len(detail_items),
                    'achieved': sum(matches),
                    'detail_texts': detail_texts
                }
    
    return subject_matches, details_matches

def analyze_vtt(vtt_content):
    try:
        captions = []
        vtt_file = StringIO(vtt_content)
        for caption in webvtt.read_buffer(vtt_file):
            captions.append(caption.text)
        
        full_text = " ".join(captions)
        
        if not full_text.strip():
            raise ValueError("VTT 파일에서 텍스트를 추출할 수 없습니다.")
        
        chunks = split_text(full_text)
        print(f"총 {len(chunks)}개의 청크로 나누어졌습니다.")
        
        analyses = []
        for i, chunk in enumerate(chunks, 1):
            print(f"VTT 청크 {i}/{len(chunks)} 분석 중...")
            analysis = analyze_text_chunk(chunk)
            if analysis:
                analyses.append(analysis)
            if i < len(chunks):
                time.sleep(10)
        
        if not analyses:
            raise ValueError("텍스트 분석에 실패했습니다.")
        
        combined_analysis = combine_analyses(analyses)
        return combined_analysis
        
    except Exception as e:
        print(f"VTT 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise

def analyze_curriculum(curriculum_content, vtt_content):
    try:
        if not curriculum_content.strip():
            raise ValueError("커리큘럼 파일이 비어있습니다.")
        
        # 커리큘럼에서 교과목 정보 추출
        topics = extract_curriculum_topics(curriculum_content)
        if not topics:
            raise ValueError("커리큘럼 분석에 실패했습니다.")
        
        # 교과목 매칭 분석
        subject_matches, details_matches = analyze_curriculum_match(vtt_content, topics)
        
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
            "details_matches": details_matches
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
    """채팅 기록을 분석하여 위험 요소를 탐지합니다."""
    try:
        system_prompt = """당신은 교육 환경에서의 채팅 기록을 전문적으로 분석하는 전문가입니다.
        채팅 내용을 깊이 있게 분석하여 전문적인 보고서 형식으로 작성해주세요.

        분석 보고서 형식:

        [채팅 분석 보고서]

        1. 위험 발언 분석
        1.1. 주요 위험 발언 요약
        • 발견된 위험 발언의 핵심 내용과 심각도를 간단히 요약
        
        1.2. 상세 위험 발언 목록
        • 구체적인 위험 발언 내용 (발언 시점과 함께)
        • 각 발언의 위험 수준 평가 (상/중/하)
        • 잠재적 영향 분석

        2. 수업 관련 피드백
        2.1. 부정적 피드백
        • 수업 내용/방식에 대한 주요 불만 사항
        • 불만의 성격 분류 (수업 내용/강의 방식/기술적 문제 등)
        
        2.2. 긍정적 피드백
        • 수업에 대한 긍정적 반응
        • 학습 효과 관련 긍정적 코멘트

        3. 상호작용 분석
        3.1. 학습 참여도
        • 학생들의 수업 참여 정도
        • 질문과 답변의 빈도
        
        3.2. 커뮤니케이션 패턴
        • 학생-교수자 간 소통 방식
        • 학생들 간의 상호작용

        4. 개선 제안
        • 발견된 문제점들에 대한 구체적 개선 방안
        • 향후 수업 운영을 위한 제언

        분석 시 주의사항:
        1. 메타데이터(발언자 ID, 타임스탬프 등)는 제외하고 실제 내용만 포함
        2. 각 발언의 맥락을 고려하여 분석
        3. 객관적이고 전문적인 용어 사용
        4. 심각한 위험 요소는 별도로 강조
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 채팅 기록을 분석해주세요:\n\n{chat_content}"}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message['content'].strip()
        
    except Exception as e:
        print(f"채팅 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'vtt_file' not in request.files or 'curriculum_file' not in request.files:
            return jsonify({'error': '파일이 누락되었습니다.'}), 400
        
        vtt_file = request.files['vtt_file']
        curriculum_file = request.files['curriculum_file']
        
        if vtt_file.filename == '' or curriculum_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        if not vtt_file.filename.endswith('.vtt'):
            return jsonify({'error': 'VTT 파일만 업로드 가능합니다.'}), 400
        
        try:
            vtt_content = vtt_file.read().decode('utf-8')
            curriculum_content = curriculum_file.read()
            
            # 파일 확장자 확인
            if curriculum_file.filename.endswith(('.xlsx', '.xls')):
                # 엑셀 파일을 JSON으로 변환
                print("Excel 파일을 JSON으로 변환 중...")
                curriculum_content = excel_to_json(curriculum_content)
            else:
                # JSON 파일인 경우 문자열로 디코딩
                curriculum_content = curriculum_content.decode('utf-8')
                # JSON 유효성 검사
                try:
                    json.loads(curriculum_content)
                except json.JSONDecodeError:
                    return jsonify({'error': '올바른 JSON 형식이 아닙니다.'}), 400
                
        except UnicodeDecodeError:
            return jsonify({'error': '파일 인코딩이 올바르지 않습니다. UTF-8 형식의 파일을 업로드해주세요.'}), 400
        
        print("VTT 파일 분석 시작...")
        vtt_analysis = analyze_vtt(vtt_content)
        print("VTT 파일 분석 완료")
        
        print("커리큘럼 파일 분석 시작...")
        curriculum_analysis = analyze_curriculum(curriculum_content, vtt_content)
        print("커리큘럼 파일 분석 완료")
        
        return jsonify({
            'progress': '분석이 완료되었습니다!',
            'vtt_analysis': vtt_analysis,
            'curriculum_analysis': curriculum_analysis
        })
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'서버 오류가 발생했습니다: {str(e)}',
            'progress': '분석 중 오류가 발생했습니다.'
        }), 500

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    try:
        if 'chat_file' not in request.files:
            return jsonify({'error': '채팅 파일이 누락되었습니다.'}), 400
        
        chat_file = request.files['chat_file']
        
        if chat_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        try:
            chat_content = chat_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({'error': '파일 인코딩이 올바르지 않습니다. UTF-8 형식의 파일을 업로드해주세요.'}), 400
        
        print("채팅 기록 분석 시작...")
        chat_analysis = analyze_chat_log(chat_content)
        print("채팅 기록 분석 완료")
        
        return jsonify({
            'progress': '채팅 분석이 완료되었습니다!',
            'chat_analysis': chat_analysis
        })
        
    except Exception as e:
        print(f"채팅 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'서버 오류가 발생했습니다: {str(e)}',
            'progress': '분석 중 오류가 발생했습니다.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 