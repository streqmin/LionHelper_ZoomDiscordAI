import os
import logging
import re
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from app.gpt_client import GPTAPIClient
import json
import queue
import threading

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'vtt'}  # 허용된 파일 확장자

# 업로드 폴더가 없으면 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API 클라이언트 초기화
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    api_client = GPTAPIClient(api_key)
    
    # API 연결 테스트
    if not api_client.test_connection():
        raise ConnectionError("API 연결 테스트 실패")
    
    logger.info("API 클라이언트 초기화 및 연결 테스트 성공")
except Exception as e:
    logger.error(f"API 클라이언트 초기화 실패: {str(e)}")
    raise

# 분석 진행 상황을 저장할 전역 큐
progress_queue = queue.Queue()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vtt_analysis')
def vtt_analysis():
    return render_template('vtt_analysis.html')

@app.route('/chat_analysis')
def chat_analysis():
    return render_template('chat_analysis.html')

@app.route('/analysis-progress')
def analysis_progress():
    def generate():
        while True:
            try:
                progress = progress_queue.get(timeout=30)  # 30초 타임아웃
                yield f"data: {json.dumps(progress)}\n\n"
            except queue.Empty:
                break
    return Response(generate(), mimetype='text/event-stream')

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    try:
        logger.info("채팅 분석 요청 수신")
        logger.info(f"요청 URL: {request.url}")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")
        
        # 채팅 파일 확인
        if 'file' not in request.files:
            logger.error("채팅 파일이 요청에 포함되지 않음")
            return jsonify({'error': '채팅 파일이 없습니다'}), 400
            
        chat_file = request.files['file']
        if chat_file.filename == '':
            logger.error("채팅 파일명이 비어있음")
            return jsonify({'error': '채팅 파일이 선택되지 않았습니다'}), 400

        # 채팅 파일 처리
        chat_filename = secure_filename(chat_file.filename)
        chat_filepath = os.path.join(app.config['UPLOAD_FOLDER'], chat_filename)
        chat_file.save(chat_filepath)
        logger.info(f"채팅 파일 저장 완료: {chat_filepath}")
        
        try:
            # 채팅 파일 내용 읽기
            with open(chat_filepath, 'r', encoding='utf-8') as f:
                chat_content = f.read()
            logger.info(f"채팅 파일 내용 읽기 성공 (길이: {len(chat_content)} 문자)")
            
            # API를 통한 분석
            chat_result = api_client.analyze_text(chat_content, 'chat')
            logger.info("채팅 분석 완료")
            
            # 결과를 HTML 형식으로 변환
            chat_html = format_analysis_result(chat_result)
            return jsonify({
                'chat_result': chat_html
            })
            
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # 임시 파일 삭제
            try:
                os.remove(chat_filepath)
                logger.info("임시 파일 삭제 완료")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
                
    except Exception as e:
        logger.error(f"요청 처리 중 예상치 못한 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_vtt', methods=['POST'])
def analyze_vtt():
    try:
        logger.info("VTT 분석 요청 수신")
        
        # 파일 처리 및 검증
        if 'vtt_file' not in request.files or 'curriculum_file' not in request.files:
            return jsonify({'error': '필요한 파일이 누락되었습니다.'}), 400
            
        vtt_file = request.files['vtt_file']
        curriculum_file = request.files['curriculum_file']
        
        if vtt_file.filename == '' or curriculum_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        # 파일 저장
        vtt_filename = secure_filename(vtt_file.filename)
        curriculum_filename = secure_filename(curriculum_file.filename)
        
        vtt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], vtt_filename)
        curriculum_filepath = os.path.join(app.config['UPLOAD_FOLDER'], curriculum_filename)
        
        vtt_file.save(vtt_filepath)
        curriculum_file.save(curriculum_filepath)
        
        try:
            # VTT 파일 내용 읽기
            with open(vtt_filepath, 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            
            # VTT 내용을 청크로 분할
            chunks = split_vtt_content(vtt_content)
            total_chunks = len(chunks)
            
            # 각 청크 분석
            analyzed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                update_progress(f"청크 {i}/{total_chunks} 분석 중")
                chunk_result = api_client.analyze_text(chunk, 'vtt')
                analyzed_chunks.append(chunk_result)
            
            update_progress("커리큘럼 매칭 분석 중")
            # 커리큘럼 파일 처리
            curriculum_content = process_curriculum_file(curriculum_filepath)
            
            # 분석 결과 통합 및 매칭
            combined_result = combine_analysis_results(analyzed_chunks)
            curriculum_result = analyze_curriculum_match(combined_result, curriculum_content)
            
            # 결과를 HTML 형식으로 변환
            vtt_html = format_analysis_result(combined_result)
            
            return jsonify({
                'vtt_result': vtt_html,
                'curriculum_result': curriculum_result
            })
            
        finally:
            # 임시 파일 삭제
            try:
                os.remove(vtt_filepath)
                os.remove(curriculum_filepath)
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
                
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_curriculum_file(filepath):
    """커리큘럼 파일(엑셀 또는 JSON)을 처리하여 내용을 반환"""
    ext = filepath.rsplit('.', 1)[1].lower()
    try:
        if ext in ['xlsx', 'xls']:
            import pandas as pd
            
            # 엑셀 파일의 모든 셀 데이터를 읽기
            df = pd.read_excel(filepath, header=None)
            
            # 결과를 저장할 리스트
            result = []
            current_subject = None
            current_details = []
            
            # 첫 번째 행이 헤더인지 확인하고 제외할 키워드 목록
            header_keywords = ['교과목명', '과목명', '교과목', '과목', 'subject']
            
            # 모든 행을 순회하면서 과목명과 세부내용 추출
            for _, row in df.iterrows():
                # 각 행의 모든 셀을 문자열로 변환하고 빈 셀 제거
                row_values = [str(cell).strip() for cell in row if str(cell).strip() != 'nan']
                if not row_values:  # 빈 행 무시
                    continue
                
                # 첫 번째 열이 비어있지 않은 경우, 새로운 과목으로 간주
                first_cell = str(row[0]).strip()
                if first_cell != 'nan' and first_cell:
                    # 헤더로 의심되는 행은 건너뛰기
                    if any(keyword.lower() in first_cell.lower() for keyword in header_keywords):
                        continue
                        
                    # 이전 과목의 정보가 있으면 저장
                    if current_subject and current_details:
                        result.append({
                            '과목명': current_subject,
                            '세부내용': current_details
                        })
                    # 새로운 과목 시작
                    current_subject = first_cell
                    current_details = []
                    # 같은 행에 세부내용이 있는 경우
                    if len(row_values) > 1:
                        current_details.extend(row_values[1:])
                else:
                    # 첫 번째 열이 비어있는 경우, 현재 과목의 세부내용으로 추가
                    if current_subject and row_values:
                        current_details.extend(row_values)
            
            # 마지막 과목 정보 추가
            if current_subject and current_details:
                result.append({
                    '과목명': current_subject,
                    '세부내용': current_details
                })
            
            if not result:
                raise ValueError('엑셀 파일에서 과목명과 세부내용을 추출할 수 없습니다.')
                
            return result
            
        elif ext == 'json':
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON 형식 검증
                if not isinstance(data, list):
                    raise ValueError('JSON 파일은 객체의 배열이어야 합니다.')
                
                result = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    
                    subject = item.get('subject') or item.get('과목명')
                    details = item.get('details') or item.get('세부내용')
                    
                    if subject and details:
                        if isinstance(details, str):
                            details = [details]
                        elif not isinstance(details, list):
                            continue
                            
                        result.append({
                            '과목명': subject,
                            '세부내용': [d for d in details if d]
                        })
                return result
        else:
            raise ValueError('지원하지 않는 파일 형식입니다')
            
    except Exception as e:
        logger.error(f"커리큘럼 파일 처리 중 오류 발생: {str(e)}")
        raise ValueError(f"커리큘럼 파일 처리 중 오류가 발생했습니다: {str(e)}")

def analyze_curriculum_match(vtt_result, curriculum_content):
    """VTT 분석 결과와 커리큘럼을 매칭하여 분석"""
    # 커리큘럼에서 과목명과 세부내용 추출
    subjects = []
    subject_details = {}
    
    for item in curriculum_content:
        subject = None
        details = None
        
        if 'subject' in item and 'details' in item:  # JSON 형식
            subject = item['subject']
            details = item['details']
        elif '과목명' in item and '세부내용' in item:  # 엑셀 형식
            subject = item['과목명']
            details = item['세부내용']
            
        if subject and details:
            if subject not in subjects:
                subjects.append(subject)
                subject_details[subject] = []
            # 리스트가 아닌 경우 리스트로 변환
            if isinstance(details, str):
                details = [details]
            subject_details[subject].extend(details)
    
    # VTT 내용 분석
    vtt_sections = vtt_result.split('---')
    vtt_content = ""
    for section in vtt_sections:
        if '주요 내용' in section or '분석' in section:
            vtt_content += section.replace('# 주요 내용', '').replace('# 분석', '')
    
    # 각 과목별 매칭 분석
    matched_subjects = []
    details_matches = {}
    
    for subject in subjects:
        # 과목별 세부내용 분석
        matched_details = []
        matches_status = []
        total_score = 0
        valid_details_count = 0
        
        # 각 세부내용에 대해 분석
        for detail in subject_details[subject]:
            if not detail or str(detail).strip() == 'nan':  # 빈 세부내용 제외
                continue
                
            valid_details_count += 1
            detail_str = str(detail).strip()
            
            # GPT API를 사용하여 세부내용과 VTT 내용의 매칭 분석
            prompt = f"""
다음 강의 내용이 특정 교과 세부내용을 다루고 있는지 분석해주세요.

[분석할 교과 세부내용]
{detail_str}

[강의 내용]
{vtt_content}

다음 형식으로 응답해주세요:
1. 달성도 (0-100): 
   - 이 강의가 해당 세부내용을 얼마나 다루었는지를 백분율로 표현
   - 직접적이고 상세한 설명이 있으면 90-100점
   - 직접적인 설명이 있으면 70-89점
   - 관련 개념이나 응용사례를 다룬 경우 50-69점
   - 간접적으로 연관된 내용을 다룬 경우 30-49점
   - 약간의 관련성만 있는 경우 10-29점
   - 매우 간접적이거나 미미한 관련성이 있는 경우 1-9점
   - 전혀 다루지 않은 경우 0점

2. 판단 근거:
   - 강의 내용 중 이 세부내용과 관련된 부분을 구체적으로 설명
   - 직접적인 언급이 없더라도 연관된 개념이나 사례가 있다면 설명
   - 매우 간접적이거나 미미한 관련성도 포함하여 설명

주의사항:
- 형식적인 단어 매칭이 아닌 실질적인 내용의 연관성을 평가해주세요
- 세부내용의 핵심 개념이나 목표가 조금이라도 다뤄졌다면 매우 관대하게 평가해주세요
- 직접적인 설명이 아니더라도, 관련 개념이나 응용 사례가 포함되어 있다면 점수를 부여해주세요
- 매우 간접적이거나 미미한 관련성이라도 발견된다면 최소 1점 이상을 부여해주세요
- 강의 내용이 해당 세부내용의 일부분만 다루더라도 그 부분에 대해 적절한 점수를 부여해주세요
"""
            
            try:
                analysis = api_client.analyze_text(prompt, 'curriculum')
                
                # 분석 결과 파싱
                lines = analysis.split('\n')
                detail_score = 0
                
                for line in lines:
                    line = line.strip()
                    if '달성도' in line:
                        try:
                            # 달성도 숫자를 더 정확하게 추출
                            score_text = line.split(':')[1].strip() if ':' in line else line
                            # 첫 번째 숫자 찾기
                            score_match = re.search(r'\d+', score_text)
                            if score_match:
                                detail_score = int(score_match.group())
                                # 점수가 100을 초과하는 경우 100으로 제한
                                detail_score = min(100, max(0, detail_score))
                                logger.info(f"추출된 달성도 점수: {detail_score} (원본 텍스트: {line})")
                        except Exception as e:
                            logger.error(f"달성도 점수 파싱 오류: {str(e)} (라인: {line})")
                            detail_score = 0
                        break
                
                # 세부내용 매칭 결과 저장
                matched_details.append(detail_str)
                matches_status.append(detail_score >= 20)  # 20% 이상이면 달성으로 판단
                total_score += detail_score
                logger.info(f"세부내용 '{detail_str}' 분석 완료 - 점수: {detail_score}")
                
            except Exception as e:
                logger.error(f"세부내용 '{detail_str}' 분석 중 오류 발생: {str(e)}")
                matched_details.append(detail_str)
                matches_status.append(False)
                total_score += 0
        
        # 과목 전체 달성도 계산
        if valid_details_count > 0:
            # 평균 점수 계산 시 소수점 아래는 버림
            achievement_rate = int(total_score / valid_details_count)
            # 최소 1%는 보장하되, 실제 점수가 있는 경우에만
            achievement_rate = max(1, achievement_rate) if total_score > 0 else 0
            logger.info(f"과목 '{subject}' 전체 달성도 계산: {achievement_rate}% (총점: {total_score}, 유효 항목 수: {valid_details_count})")
        else:
            achievement_rate = 0
            logger.info(f"과목 '{subject}'의 유효한 세부내용이 없음")
        
        matched_subjects.append({
            'name': subject,
            'achievement_rate': achievement_rate
        })
        
        details_matches[subject] = {
            'matches': matches_status,
            'detail_texts': matched_details
        }
    
    return {
        'matched_subjects': matched_subjects,
        'details_matches': details_matches
    }

def summarize_content(content_list, max_length=800):
    """여러 내용을 하나로 통합하여 재요약"""
    if not content_list:
        return []
        
    # 모든 내용을 하나의 문자열로 결합
    combined_content = "\n".join(content_list)
    
    try:
        # GPT API를 통해 재요약
        prompt = f"""다음 내용을 {max_length}자 이내로 통합하여 요약해주세요. 
        중요한 내용을 놓치지 않되, 반복되는 내용은 제거하고 핵심적인 내용만 남겨주세요.
        각 요점은 새로운 줄에 '- '로 시작하도록 해주세요.
        
        내용:
        {combined_content}"""
        
        summarized = api_client.analyze_text(prompt, 'summarize')
        # 결과를 리스트로 변환
        return [line.strip()[2:] for line in summarized.split('\n') if line.strip().startswith('- ')]
    except Exception as e:
        logger.error(f"재요약 중 오류 발생: {str(e)}")
        return content_list  # 오류 발생 시 원본 내용 반환

def format_analysis_result(content):
    """분석 결과를 HTML 형식으로 변환"""
    logger.info(f"원본 분석 결과: {content}")
    
    # 섹션을 분리 (--- 구분자 기준)
    sections = content.split('---')
    logger.info(f"섹션 분할 결과: {sections}")
    
    # 카테고리별로 내용을 저장할 딕셔너리
    categories = {
        '주요 대화 주제': [],
        '수강생 감정/태도 분석': {
            '1. 긍정적 반응': [],
            '2. 부정적 반응': [],
            '3. 질문/요청사항': []
        },
        '어려움/불만 상세 분석': {
            '1. 학습적 어려움': [],
            '2. 수업 진행 관련 문제': [],
            '3. 기술적 문제': []
        },
        '개선 제안': {
            '1. 학습 내용 개선': [],
            '2. 수업 방식 개선': [],
            '3. 기술적 지원 강화': []
        },
        '위험 발언 및 주의사항': [],
        '종합 제언': []
    }
    
    # 모든 섹션의 내용을 카테고리별로 분류
    for section in sections:
        if not section.strip():
            continue
            
        logger.info(f"처리 중인 섹션: {section}")
        
        # 각 섹션의 내용을 파싱
        lines = section.strip().split('\n')
        current_category = None
        current_subcategory = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                current_category = line[2:].strip()  # '#' 제거
                current_subcategory = None
                continue
                
            # 하위 카테고리가 있는 섹션 처리
            if current_category in ['수강생 감정/태도 분석', '어려움/불만 상세 분석', '개선 제안']:
                subcategories = categories[current_category]
                for subcat in subcategories.keys():
                    if line.startswith(subcat):
                        current_subcategory = subcat
                        break
                if current_subcategory and line.startswith('- '):
                    # 중복 제거를 위해 이미 있는 항목은 추가하지 않음
                    if line not in categories[current_category][current_subcategory]:
                        categories[current_category][current_subcategory].append(line)
                    continue
            
            # 다른 카테고리 처리
            if current_category in categories and isinstance(categories[current_category], list):
                if line not in categories[current_category]:  # 중복 제거
                    categories[current_category].append(line)
    
    # HTML 생성
    html_content = ['<div class="analysis-result">']
    
    # 주요 대화 주제 섹션
    if categories['주요 대화 주제']:
        # 주요 대화 주제를 하나의 문단으로 합치기
        main_topics = []
        for item in categories['주요 대화 주제']:
            if item.startswith('- '):
                main_topics.append(item[2:].strip())
            else:
                main_topics.append(item.strip())
        
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">주요 대화 주제</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(main_topics)}</p>',
            '    </div>',
            '</div>'
        ])
    
    # 수강생 감정/태도 분석 섹션
    if any(categories['수강생 감정/태도 분석'].values()):
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">수강생 감정/태도 분석</h2>'
        ])
        
        for subcategory, items in categories['수강생 감정/태도 분석'].items():
            if items:  # 해당 하위 카테고리에 내용이 있는 경우에만 표시
                html_content.extend([
                    f'    <div class="subsection">',
                    f'        <h3 class="subsection-title">{subcategory}</h3>',
                    f'        <ul class="analysis-list">'
                ])
                for item in items:
                    html_content.append(f'            <li>{item[2:]}</li>')  # '- ' 제거
                html_content.extend([
                    '        </ul>',
                    '    </div>'
                ])
        
        html_content.append('</div>')
    
    # 어려움/불만 상세 분석 섹션
    if any(categories['어려움/불만 상세 분석'].values()):
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">어려움/불만 상세 분석</h2>'
        ])
        
        for subcategory, items in categories['어려움/불만 상세 분석'].items():
            if items:  # 해당 하위 카테고리에 내용이 있는 경우에만 표시
                html_content.extend([
                    f'    <div class="subsection">',
                    f'        <h3 class="subsection-title">{subcategory}</h3>',
                    f'        <ul class="analysis-list">'
                ])
                for item in items:
                    html_content.append(f'            <li>{item[2:]}</li>')  # '- ' 제거
                html_content.extend([
                    '        </ul>',
                    '    </div>'
                ])
        
        html_content.append('</div>')

    # 개선 제안 섹션
    if any(categories['개선 제안'].values()):
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">개선 제안</h2>'
        ])
        
        for subcategory, items in categories['개선 제안'].items():
            if items:  # 해당 하위 카테고리에 내용이 있는 경우에만 표시
                html_content.extend([
                    f'    <div class="subsection">',
                    f'        <h3 class="subsection-title">{subcategory}</h3>',
                    f'        <ul class="analysis-list">'
                ])
                for item in items:
                    html_content.append(f'            <li>{item[2:]}</li>')  # '- ' 제거
                html_content.extend([
                    '        </ul>',
                    '    </div>'
                ])
        
        html_content.append('</div>')

    # 위험 발언 및 주의사항 섹션
    risk_items = []
    has_real_risks = False
    
    for item in categories['위험 발언 및 주의사항']:
        item = item.strip()
        # 위험 발언이 없다는 내용의 텍스트는 제외
        if (item and 
            not item.endswith('없습니다.') and 
            not item.startswith('특별한 주의사항이 없') and
            not '발견되지 않' in item and
            not '확인되지 않' in item and
            not '포함되어 있지 않' in item and
            not '위험한 내용이 없' in item and
            not '특별한 위험' in item and
            not '부적절한 내용이 없' in item):
            risk_items.append(item)
            has_real_risks = True
    
    if has_real_risks and risk_items:  # 실제 위험 발언이 있는 경우에만
        html_content.extend([
            '<div class="category-section risk-section">',
            '    <h2 class="category-title">위험 발언 및 주의사항</h2>',
            '    <div class="risk-summary">',
            '        <div class="risk-icon">⚠️</div>',
            '        <p>채팅에서 다음과 같은 위험 발언이 감지되었습니다.</p>',
            '    </div>',
            '    <ul class="risk-list">'
        ])
        for item in risk_items:
            html_content.append(f'        <li>{item}</li>')
        html_content.extend([
            '    </ul>',
            '</div>'
        ])
    elif any(categories.values()):  # 다른 카테고리에 내용이 있는 경우에만
        # 위험 발언이 없는 경우
        html_content.extend([
            '<div class="category-section risk-section safe">',
            '    <h2 class="category-title">위험 발언 및 주의사항</h2>',
            '    <div class="risk-summary">',
            '        <div class="risk-icon">✅</div>',
            '        <p>채팅에서 특별한 위험 발언이 감지되지 않았습니다.</p>',
            '    </div>',
            '</div>'
        ])

    # 종합 제언 섹션
    if categories['종합 제언']:
        # 종합 제언을 하나의 문단으로 합치기
        recommendations = []
        for item in categories['종합 제언']:
            if item.startswith('- '):
                recommendations.append(item[2:].strip())
            else:
                recommendations.append(item.strip())
        
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">종합 제언</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(recommendations)}</p>',
            '    </div>',
            '</div>'
        ])
    
    html_content.append('</div>')
    return '\n'.join(html_content)

def format_list_items(content):
    """목록 항목을 HTML 형식으로 변환"""
    items = []
    for line in content.split(chr(10)):  # chr(10)은 '\n'과 동일
        line = line.strip()
        if line.startswith('- '):
            items.append(f'<li>{line[2:].strip()}</li>')
        elif line.startswith('• '):
            items.append(f'<li>{line[2:].strip()}</li>')
        elif line:  # 일반 텍스트인 경우
            items.append(f'<li>{line}</li>')
    return '\n'.join(items)

def update_progress(message):
    """분석 진행 상황을 큐에 추가"""
    progress_queue.put({'message': message})

def split_vtt_content(content, chunk_size=5000):
    """VTT 내용을 청크로 분할"""
    words = content.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # 공백 포함
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def combine_analysis_results(results):
    """여러 청크의 분석 결과를 하나로 통합"""
    combined = {
        '주요 내용': [],
        '키워드': set(),
        '분석': [],
        '위험 발언': []
    }
    
    for result in results:
        sections = result.split('---')
        current_category = None
        
        for section in sections:
            lines = section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('# '):
                    current_category = line[2:].strip()
                    continue
                if line and current_category in combined:
                    if current_category == '키워드':
                        combined[current_category].update(line.split(', '))
                    else:
                        combined[current_category].append(line)
    
    # 키워드를 리스트로 변환하고 정렬
    combined['키워드'] = sorted(list(combined['키워드']))
    
    # 결과를 문자열로 변환
    return '\n---\n'.join([
        f"# {category}\n" + '\n'.join(items if isinstance(items, list) else [items])
        for category, items in combined.items()
    ])

if __name__ == '__main__':
    app.run(debug=True) 