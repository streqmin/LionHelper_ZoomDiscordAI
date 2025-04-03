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

@app.route('/analyze', methods=['POST'])
@app.route('/analyze_chat', methods=['POST'])
@app.route('/analyze_vtt', methods=['POST'])
def analyze():
    try:
        logger.info("분석 요청 수신")
        logger.info(f"요청 URL: {request.url}")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")
        
        # VTT 파일 확인
        if 'vtt_file' not in request.files:
            logger.error("VTT 파일이 요청에 포함되지 않음")
            return jsonify({'error': 'VTT 파일이 없습니다'}), 400
            
        vtt_file = request.files['vtt_file']
        if vtt_file.filename == '':
            logger.error("VTT 파일명이 비어있음")
            return jsonify({'error': 'VTT 파일이 선택되지 않았습니다'}), 400

        # 커리큘럼 파일 확인
        if 'curriculum_file' not in request.files:
            logger.error("커리큘럼 파일이 요청에 포함되지 않음")
            return jsonify({'error': '커리큘럼 파일이 없습니다'}), 400
            
        curriculum_file = request.files['curriculum_file']
        if curriculum_file.filename == '':
            logger.error("커리큘럼 파일명이 비어있음")
            return jsonify({'error': '커리큘럼 파일이 선택되지 않았습니다'}), 400

        # VTT 파일 처리
        vtt_filename = secure_filename(vtt_file.filename)
        vtt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], vtt_filename)
        vtt_file.save(vtt_filepath)
        logger.info(f"VTT 파일 저장 완료: {vtt_filepath}")

        # 커리큘럼 파일 처리
        curriculum_filename = secure_filename(curriculum_file.filename)
        curriculum_filepath = os.path.join(app.config['UPLOAD_FOLDER'], curriculum_filename)
        curriculum_file.save(curriculum_filepath)
        logger.info(f"커리큘럼 파일 저장 완료: {curriculum_filepath}")
        
        try:
            # VTT 파일 내용 읽기
            with open(vtt_filepath, 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            logger.info(f"VTT 파일 내용 읽기 성공 (길이: {len(vtt_content)} 문자)")
            
            # 커리큘럼 파일 처리 (엑셀 또는 JSON)
            curriculum_content = process_curriculum_file(curriculum_filepath)
            logger.info("커리큘럼 파일 처리 완료")
            
            # API를 통한 분석
            vtt_result = api_client.analyze_text(vtt_content, 'vtt')
            curriculum_result = analyze_curriculum_match(vtt_result, curriculum_content)
            logger.info("분석 완료")
            
            # 결과를 HTML 형식으로 변환
            vtt_html = format_analysis_result(vtt_result)
            return jsonify({
                'vtt_result': vtt_html,
                'curriculum_result': curriculum_result
            })
            
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # 임시 파일 삭제
            try:
                os.remove(vtt_filepath)
                os.remove(curriculum_filepath)
                logger.info("임시 파일 삭제 완료")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
                
    except Exception as e:
        logger.error(f"요청 처리 중 예상치 못한 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_curriculum_file(filepath):
    """커리큘럼 파일(엑셀 또는 JSON)을 처리하여 내용을 반환"""
    ext = filepath.rsplit('.', 1)[1].lower()
    try:
        if ext in ['xlsx', 'xls']:
            import pandas as pd
            df = pd.read_excel(filepath)
            
            # 필수 컬럼 확인
            required_columns = ['과목명', '세부내용']
            if not all(col in df.columns for col in required_columns):
                raise ValueError('엑셀 파일에 필수 컬럼(과목명, 세부내용)이 없습니다.')
            
            # NaN 값을 빈 문자열로 변환
            df = df.fillna('')
            
            # 과목별로 세부내용 정리
            result = []
            for subject in df['과목명'].unique():
                if subject:  # 과목명이 비어있지 않은 경우만
                    details = df[df['과목명'] == subject]['세부내용'].tolist()
                    # 빈 문자열 제외
                    details = [d for d in details if d]
                    if details:  # 세부내용이 있는 경우만
                        result.append({
                            '과목명': subject,
                            '세부내용': details
                        })
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
            subject_details[subject].append(details)
    
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
        # 과목별 매칭 점수 계산
        subject_score = 0
        total_details = len(subject_details[subject])
        matched_details = []
        matches_status = []
        
        # GPT API를 사용하여 각 세부내용과 VTT 내용의 매칭 분석
        prompt = f"""
다음 강의 내용이 특정 교과목의 세부내용들과 얼마나 일치하는지 분석해주세요.

교과목: {subject}
세부내용:
{chr(10).join(f"- {detail}" for detail in subject_details[subject])}

강의 내용:
{vtt_content}

각 세부내용별로 강의 내용과의 일치 여부를 판단하고, 일치하는 경우 관련된 부분을 구체적으로 설명해주세요.
응답 형식:
1. 전체 일치도: (0-100 사이의 숫자)
2. 세부내용 분석:
- [일치/불일치] (세부내용): (구체적인 설명)
"""
        
        try:
            analysis = api_client.analyze_text(prompt, 'curriculum')
            
            # 분석 결과 파싱
            lines = analysis.split('\n')
            achievement_rate = 0
            
            for line in lines:
                if line.startswith('1. 전체 일치도:'):
                    try:
                        achievement_rate = int(re.search(r'\d+', line).group())
                    except:
                        achievement_rate = 0
                elif line.startswith('- [일치]') or line.startswith('- [불일치]'):
                    is_match = line.startswith('- [일치]')
                    detail_text = line[line.find(']')+1:].strip()
                    if ':' in detail_text:
                        detail_text = detail_text.split(':', 1)[0].strip()
                    matched_details.append(detail_text)
                    matches_status.append(is_match)
        
        except Exception as e:
            logger.error(f"과목 {subject} 분석 중 오류 발생: {str(e)}")
            achievement_rate = 0
            matched_details = subject_details[subject]
            matches_status = [False] * len(matched_details)
        
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
        '주요 내용': [],
        '키워드': [],
        '분석': [],
        '위험 발언': []
    }
    
    # 모든 섹션의 내용을 카테고리별로 분류
    for section in sections:
        if not section.strip():
            continue
            
        logger.info(f"처리 중인 섹션: {section}")
        
        # 각 섹션의 내용을 파싱
        lines = section.strip().split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                current_category = line[2:].strip()  # '#' 제거
                continue
            if line and current_category in categories:
                categories[current_category].append(line)
    
    # 주요 내용과 분석 섹션 재요약
    categories['주요 내용'] = summarize_content(categories['주요 내용'])
    categories['분석'] = summarize_content(categories['분석'])
    
    # HTML 생성
    html_content = ['<div class="analysis-result">']
    
    # 주요 내용 섹션
    if categories['주요 내용']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">주요 내용</h2>',
            '    <ul class="content-list">'
        ])
        for item in categories['주요 내용']:
            html_content.append(f'        <li>{item}</li>')
        html_content.extend([
            '    </ul>',
            '</div>'
        ])
    
    # 키워드 섹션
    if categories['키워드']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">키워드</h2>',
            '    <ul class="keyword-list">'
        ])
        for item in categories['키워드']:
            html_content.append(f'        <li>{item}</li>')
        html_content.extend([
            '    </ul>',
            '</div>'
        ])
    
    # 분석 섹션
    if categories['분석']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">분석</h2>',
            '    <ul class="analysis-list">'
        ])
        for item in categories['분석']:
            html_content.append(f'        <li>{item}</li>')
        html_content.extend([
            '    </ul>',
            '</div>'
        ])

    # 위험 발언 섹션
    # 모든 섹션의 위험 발언을 검사하여 실제 위험 발언이 있는지 확인
    risk_items = []
    has_real_risks = False
    
    for item in categories['위험 발언']:
        item = item.strip()
        # 위험 발언이 없다는 내용의 텍스트는 제외
        if (item and 
            not item.endswith('없습니다.') and 
            not item.startswith('위험 발언이 없') and
            not item.startswith('- 위험 발언이 없') and
            not '발견되지 않' in item and
            not '확인되지 않' in item and
            not '포함되어 있지 않' in item and
            not '위험한 내용이 없' in item and
            not '특별한 위험' in item and
            not '부적절한 내용이 없' in item):
            # 실제 위험 발언인 경우에만 추가
            if not any(safe_phrase in item.lower() for safe_phrase in [
                '없습니다', '발견되지 않', '확인되지 않', '포함되어 있지 않',
                '감지되지 않', '발견할 수 없', '문제가 없'
            ]):
                risk_items.append(item)
                has_real_risks = True
    
    if has_real_risks and risk_items:  # 실제 위험 발언이 있는 경우에만
        html_content.extend([
            '<div class="category-section risk-section">',
            '    <h2 class="category-title">위험 발언 분석</h2>',
            '    <div class="risk-summary">',
            '        <div class="risk-icon">⚠️</div>',
            '        <p>강의 중 다음과 같은 위험 발언이 감지되었습니다.</p>',
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
            '    <h2 class="category-title">위험 발언 분석</h2>',
            '    <div class="risk-summary">',
            '        <div class="risk-icon">✅</div>',
            '        <p>강의에서 특별한 위험 발언이 감지되지 않았습니다.</p>',
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