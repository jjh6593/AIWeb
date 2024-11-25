from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
import json
from sklearn.linear_model import LinearRegression  # 예시로 LinearRegression 사용
# CORS 설정을 위해 필요하다면 다음 코드를 추가하세요.
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
# 업로드 및 모델 저장 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)



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

@app.route('/api/save_filtered_csv', methods=['POST'])
def save_filtered_csv():
    data = request.json
    exclude_columns = data.get('exclude_columns', [])
    new_filename = data.get('new_filename', '')
    filename = data.get('filename', '')  # JSON에서 filename 가져오기

    if not new_filename:
        return jsonify({'status': 'error', 'message': '새 파일 이름이 필요합니다.'}), 400

    if not filename:
        return jsonify({'status': 'error', 'message': '원본 파일명이 필요합니다.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # current_app 대신 app 사용
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': '원본 파일을 찾을 수 없습니다.'}), 404

    # CSV 파일에서 제외된 컬럼을 제거하고 저장
    df = pd.read_csv(file_path)
    filtered_df = df.drop(columns=exclude_columns, errors='ignore')
    new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    filtered_df.to_csv(new_file_path, index=False)

    return jsonify({'status': 'success', 'message': f'필터링된 데이터가 {new_filename}로 저장되었습니다.'})

# 모델 생성 및 저장
@app.route('/api/save_model', methods=['POST'])
def save_model():
    data = request.json
    model_type = data.get('model_type')
    model_name = data.get('model_name')
    model_selected = data.get('model_selected')

    if not model_type or not model_name or not model_selected:
        return jsonify({'status': 'error', 'message': '모델 타입, 이름, 그리고 모델을 선택하세요.'}), 400

    # 모델 정보 저장
    model_info = {
        'model_type': model_type,
        'model_name': model_name,
        'model_selected': model_selected,
        'parameters': {}  # 추후에 파라미터 추가 가능
    }
    model_file = os.path.join(MODEL_FOLDER, f"{model_name}.json")
    with open(model_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    return jsonify({'status': 'success', 'message': f'모델 {model_name}이(가) 저장되었습니다.'})

# 저장된 모델 목록 가져오기
@app.route('/api/get_models', methods=['GET'])
def get_models():
    models = []
    for filename in os.listdir(MODEL_FOLDER):
        if filename.endswith('.json'):
            with open(os.path.join(MODEL_FOLDER, filename), 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                models.append(model_data)
    return jsonify({'status': 'success', 'models': models})

# 모델 삭제
@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    data = request.json
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({'status': 'error', 'message': '모델 이름이 필요합니다.'}), 400

    model_file = os.path.join(MODEL_FOLDER, f"{model_name}.json")
    if os.path.exists(model_file):
        os.remove(model_file)
        return jsonify({'status': 'success', 'message': f'모델 {model_name}이(가) 삭제되었습니다.'})
    else:
        return jsonify({'status': 'error', 'message': f'모델 {model_name}을(를) 찾을 수 없습니다.'}), 404

# 모델 업로드 API
@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files or 'model_name' not in request.form:
        return jsonify({'status': 'error', 'message': '모델 파일과 이름이 필요합니다.'}), 400

    model_file = request.files['model_file']
    model_name = request.form['model_name']

    if model_file.filename == '':
        return jsonify({'status': 'error', 'message': '파일이 선택되지 않았습니다.'}), 400

    # 허용된 파일 확장자 확인
    ALLOWED_EXTENSIONS = {'pkl', 'pt', 'h5'}
    if not ('.' in model_file.filename and model_file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
        return jsonify({'status': 'error', 'message': '허용되지 않는 파일 형식입니다.'}), 400

    # 모델 파일 저장
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.{model_file.filename.rsplit('.', 1)[1].lower()}")
    model_file.save(model_path)

    # 모델 정보 저장
    model_info = {
        'model_name': model_name,
        'file_path': model_path,
        'type': 'uploaded'
    }
    metadata_path = os.path.join(MODEL_FOLDER, f"{model_name}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    return jsonify({'status': 'success', 'message': f'모델 {model_name}이(가) 업로드되었습니다.'})


if __name__ == '__main__':
    app.run(debug=True)