import os
import logging
import re
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from app.gpt_client import GPTAPIClient

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
    if ext in ['xlsx', 'xls']:
        import pandas as pd
        df = pd.read_excel(filepath)
        return df.to_dict('records')
    elif ext == 'json':
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError('지원하지 않는 파일 형식입니다')

def analyze_curriculum_match(vtt_result, curriculum_content):
    """VTT 분석 결과와 커리큘럼을 매칭하여 분석"""
    # TODO: 실제 매칭 로직 구현
    return {
        'summary': '커리큘럼 매칭 분석 결과입니다.',
        'matched_subjects': [
            {'name': '과목 1', 'achievement_rate': 80},
            {'name': '과목 2', 'achievement_rate': 60}
        ],
        'details_matches': {
            '과목 1': {
                'matches': [True, False],
                'detail_texts': ['세부내용 1', '세부내용 2']
            },
            '과목 2': {
                'matches': [True],
                'detail_texts': ['세부내용 1']
            }
        }
    }

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
    for item in categories['위험 발언']:
        item = item.strip()
        # 위험 발언이 없다는 내용의 텍스트는 제외
        if (item and 
            not item.endswith('없습니다.') and 
            not item.startswith('위험 발언이 없') and
            not item.startswith('- 위험 발언이 없') and
            not '발견되지 않' in item and
            not '확인되지 않' in item and
            not '포함되어 있지 않' in item):
            risk_items.append(item)
    
    if risk_items:
        # 위험 발언이 있는 경우
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

if __name__ == '__main__':
    app.run(debug=True) 