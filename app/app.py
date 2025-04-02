import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from app.config import Config
from app.tasks import celery, analyze_vtt_task

app = Flask(__name__)
app.config.from_object(Config)

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 비동기 분석 작업 시작
        task = analyze_vtt_task.delay(file_path)
        
        return jsonify({
            'message': '분석이 시작되었습니다.',
            'task_id': task.id
        })
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

@app.route('/status/<task_id>')
def get_status(task_id):
    task = analyze_vtt_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': '작업이 대기 중입니다...'
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'error': str(task.result)
        }
    return jsonify(response)

@app.route('/result/<filename>')
def get_result(filename):
    result_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(result_file):
        return send_file(result_file, as_attachment=True)
    return jsonify({'error': '결과 파일을 찾을 수 없습니다.'}), 404

if __name__ == '__main__':
    app.run(debug=True) 