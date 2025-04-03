import os
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from werkzeug.utils import secure_filename
from app.config import Config
from datetime import datetime
import json
import logging
from .simple_client import SimpleAPIClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
app.config['ALLOWED_EXTENSIONS'] = {'vtt', 'txt'}  # 허용된 파일 확장자

# 업로드 폴더가 없으면 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API 클라이언트 초기화
try:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
    api_client = SimpleAPIClient(api_key)
    logger.info("API 클라이언트 초기화 성공")
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
def analyze():
    try:
        logger.info("분석 요청 수신")
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
        
        # 파일 내용 읽기
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"파일 내용 읽기 성공. 길이: {len(content)} 문자")
        except Exception as e:
            logger.error(f"파일 읽기 실패: {str(e)}")
            return jsonify({'error': f'파일 읽기 실패: {str(e)}'}), 500
        
        # 분석 유형 결정
        analysis_type = 'chat' if 'chat' in filename.lower() else 'vtt'
        logger.info(f"분석 유형 결정: {analysis_type}")
        
        # API를 통한 분석
        try:
            logger.info("API 분석 시작")
            result = api_client.analyze_text(content, analysis_type)
            logger.info("API 분석 완료")
            
            if not result:
                logger.error("API 분석 결과가 비어있음")
                return jsonify({'error': '분석 결과가 비어있습니다'}), 500
                
        except Exception as e:
            logger.error(f"API 분석 중 오류 발생: {str(e)}")
            return jsonify({'error': f'분석 중 오류 발생: {str(e)}'}), 500
        finally:
            # 임시 파일 삭제
            try:
                os.remove(filepath)
                logger.info("임시 파일 삭제 완료")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
        
        return jsonify({'result': result})
        
    except Exception as e:
        logger.error(f"처리 중 예상치 못한 오류 발생: {str(e)}")
        return jsonify({'error': f'처리 중 오류 발생: {str(e)}'}), 500

@app.route('/result/<filename>')
def get_result(filename):
    result_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(result_file):
        return send_file(result_file, as_attachment=True)
    return jsonify({'error': '결과 파일을 찾을 수 없습니다.'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 