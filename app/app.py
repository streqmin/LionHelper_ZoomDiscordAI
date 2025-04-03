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
        
        if 'file' not in request.files:
            logger.error("파일이 요청에 포함되지 않음")
            return jsonify({'error': '파일이 없습니다'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("파일명이 비어있음")
            return jsonify({'error': '선택된 파일이 없습니다'}), 400
            
        if not allowed_file(file.filename):
            logger.error(f"허용되지 않은 파일 형식: {file.filename}")
            return jsonify({'error': '허용되지 않은 파일 형식입니다'}), 400
            
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"파일 저장 완료: {filepath}")
        
        try:
            # 파일 내용 읽기
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"파일 내용 읽기 성공 (길이: {len(content)} 문자)")
            
            # 분석 유형 결정
            analysis_type = 'chat' if 'chat' in filename.lower() else 'vtt'
            logger.info(f"분석 유형 결정: {analysis_type}")
            
            # API를 통한 분석
            result = api_client.analyze_text(content, analysis_type)
            logger.info("분석 완료")
            
            # 결과를 HTML 형식으로 변환
            html_result = format_analysis_result(result)
            return jsonify({'result': html_result})
            
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # 임시 파일 삭제
            try:
                os.remove(filepath)
                logger.info("임시 파일 삭제 완료")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
                
    except Exception as e:
        logger.error(f"요청 처리 중 예상치 못한 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

def format_analysis_result(content):
    """분석 결과를 HTML 형식으로 변환"""
    logger.info(f"원본 분석 결과: {content}")
    
    sections = re.split(r'(?=\d\. )', content)
    logger.info(f"섹션 분할 결과: {sections}")
    
    html_content = ''
    
    for section in sections:
        if section.strip():
            logger.info(f"처리 중인 섹션: {section}")
            main_match = re.match(r'^(\d+)\. (.+?)(?:\n|$)', section, re.MULTILINE)
            
            if not main_match:
                logger.warning(f"메인 섹션 매칭 실패: {section}")
                continue
                
            main_number = main_match.group(1)
            main_title = main_match.group(2)
            logger.info(f"메인 섹션 매칭: 번호={main_number}, 제목={main_title}")
            
            section_content = re.sub(r'^\d+\. .+?(?:\n|$)', '', section, flags=re.MULTILINE).strip()
            logger.info(f"섹션 내용: {section_content}")
            
            subsections = re.split(r'(?=\d+\.\d+ )', section_content)
            logger.info(f"하위 섹션 분할: {subsections}")
            
            html_content += f'''
                <div class="summary-section">
                    <h2><span class="section-number">{main_number}.</span>{main_title}</h2>
                    {format_subsections(subsections)}
                </div>
            '''
    
    logger.info(f"최종 HTML 결과: {html_content}")
    return html_content

def format_subsections(subsections):
    """하위 섹션을 HTML 형식으로 변환"""
    html_subsections = ''
    
    for subsection in subsections:
        if not subsection.strip():
            continue
            
        logger.info(f"처리 중인 하위 섹션: {subsection}")
        sub_match = re.match(r'^(\d+\.\d+) (.+?)(?:\n|$)', subsection, re.MULTILINE)
        
        if not sub_match:
            logger.warning(f"하위 섹션 매칭 실패: {subsection}")
            continue
            
        sub_number = sub_match.group(1)
        sub_title = sub_match.group(2)
        logger.info(f"하위 섹션 매칭: 번호={sub_number}, 제목={sub_title}")
        
        sub_content = re.sub(r'^\d+\.\d+ .+?(?:\n|$)', '', subsection, flags=re.MULTILINE).strip()
        logger.info(f"하위 섹션 내용: {sub_content}")
        
        html_subsections += f'''
            <div class="subsection">
                <div class="subsection-title">
                    <span class="section-number">{sub_number}</span>{sub_title}
                </div>
                <ul>
                    {format_list_items(sub_content)}
                </ul>
            </div>
        '''
    
    return html_subsections

def format_list_items(content):
    """목록 항목을 HTML 형식으로 변환"""
    items = content.split('\n')
    html_items = ''
    
    for item in items:
        item = item.strip()
        if item.startswith('•') or item.startswith('-'):
            html_items += f'<li>{item[1:].strip()}</li>'
    
    return html_items

if __name__ == '__main__':
    app.run(debug=True) 