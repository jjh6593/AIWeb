from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
import json
from sklearn.linear_model import LinearRegression  # 예시로 LinearRegression 사용

app = Flask(__name__)

# 업로드 및 모델 저장 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# CORS 설정을 위해 필요하다면 다음 코드를 추가하세요.
from flask_cors import CORS
CORS(app)

# 라우트 설정

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static_file(path):
    return send_from_directory('static', path)

# API 엔드포인트

# CSV 파일 업로드
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '파일이 없습니다.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '파일이 선택되지 않았습니다.'}), 400
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    return jsonify({'status': 'success', 'message': '파일 업로드 성공', 'filename': filename})

# 업로드된 CSV 파일 목록 가져오기
@app.route('/api/get_csv_files', methods=['GET'])
def get_csv_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({'status': 'success', 'files': files})

# CSV 파일 내용 가져오기
@app.route('/api/get_csv_data', methods=['GET'])
def get_csv_data():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'status': 'error', 'message': '파일명이 제공되지 않았습니다.'}), 400
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': '파일을 찾을 수 없습니다.'}), 404
    df = pd.read_csv(file_path)
    data_preview = df.head().to_dict(orient='records')
    columns = df.columns.tolist()
    return jsonify({'status': 'success', 'data_preview': data_preview, 'columns': columns})

# 모델 생성 및 저장
@app.route('/api/save_model', methods=['POST'])
def save_model():
    model_info = request.json
    model_name = model_info.get('model_name')
    if not model_name:
        return jsonify({'status': 'error', 'message': '모델 이름이 필요합니다.'}), 400
    model_path = os.path.join(MODEL_FOLDER, f'{model_name}.json')
    with open(model_path, 'w') as f:
        json.dump(model_info, f)
    return jsonify({'status': 'success', 'message': '모델이 저장되었습니다.'})

# 저장된 모델 목록 가져오기
@app.route('/api/get_models', methods=['GET'])
def get_models():
    files = [f[:-5] for f in os.listdir(MODEL_FOLDER) if f.endswith('.json')]
    return jsonify({'status': 'success', 'models': files})

# 모델 학습
@app.route('/api/train_model', methods=['POST'])
def train_model():
    data = request.json
    csv_filename = data.get('csv_filename')
    model_name = data.get('model_name')
    target_column = data.get('target_column')

    if not csv_filename or not model_name or not target_column:
        return jsonify({'status': 'error', 'message': '필요한 정보가 부족합니다.'}), 400

    # CSV 파일 로드
    csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': 'CSV 파일을 찾을 수 없습니다.'}), 404
    df = pd.read_csv(csv_path)

    # 모델 정보 로드
    model_path = os.path.join(MODEL_FOLDER, f'{model_name}.json')
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': '모델을 찾을 수 없습니다.'}), 404
    with open(model_path, 'r') as f:
        model_info = json.load(f)

    # 데이터 전처리 및 학습 (예시로 간단하게 구현)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 간단한 LinearRegression 모델로 학습 (실제 구현에서는 다양한 모델 적용)
    model = LinearRegression()
    model.fit(X, y)

    # 학습 결과 저장 (여기서는 단순히 score를 계산)
    score = model.score(X, y)
    result = {
        'model_name': model_name,
        'score': score
    }

    # 결과를 JSON 파일로 저장
    result_path = os.path.join(MODEL_FOLDER, f'{model_name}_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f)

    return jsonify({'status': 'success', 'message': '모델 학습 완료', 'result': result})

# 학습 결과 가져오기
@app.route('/api/get_training_results', methods=['GET'])
def get_training_results():
    results = []
    for file in os.listdir(MODEL_FOLDER):
        if file.endswith('_result.json'):
            with open(os.path.join(MODEL_FOLDER, file), 'r') as f:
                result = json.load(f)
                results.append(result)
    return jsonify({'status': 'success', 'results': results})

if __name__ == '__main__':
    app.run(debug=True)
