import os
import logging
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
            
            return jsonify({'result': result})
            
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

if __name__ == '__main__':
    app.run(debug=True) 