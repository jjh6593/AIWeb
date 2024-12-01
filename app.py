from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import pandas as pd
import json
from model import create_model, MLP
from data_preprocessing import MinMaxScaling
import joblib
from data_utils import load_data, save_data, preprocess_data, get_columns, get_data_preview
# 전역에서 PyTorch와 관련 모듈 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from train import train_pytorch_model, train_sklearn_model

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

# 1 CSV 파일 업로드
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

# 2 업로드된 CSV 파일 목록 가져오기
@app.route('/api/get_csv_files', methods=['GET'])
def get_csv_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({'status': 'success', 'files': files})

# 3 CSV 파일 내용 가져오기
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

# 4. 필터링된 CSV 저장
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

# 5. 데이터 설정 제출
@app.route('/api/submit_data_settings', methods=['POST'])
def submit_data_settings():
    filename = request.form.get('filename')
    target_column = request.form.get('targetColumn')
    scaler = request.form.get('scaler')
    missing_handling = request.form.get('missingHandling')

    if not filename or not target_column:
        return jsonify({'status': 'error', 'message': '필수 정보가 제공되지 않았습니다.'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(file_path)

    # 데이터 전처리 및 저장 (필요에 따라 추가 구현 가능)
    X, y = preprocess_data(df, target_column, scaler, missing_handling)

    # 전처리된 데이터를 저장하거나 세션에 저장하는 등의 처리가 필요할 수 있습니다.

    return jsonify({'status': 'success', 'message': '데이터 설정이 완료되었습니다.'})


@app.route('/api/save_model', methods=['POST'])
def save_model():
    data = request.json
    model_type = data.get('model_type')
    model_name = data.get('model_name')
    model_selected = data.get('model_selected')
    input_size = data.get('input_size', None)

    if not model_type or not model_name or not model_selected:
        return jsonify({'status': 'error', 'message': '모델 타입, 이름, 그리고 모델을 선택하세요.'}), 400

    # 모델 생성
    try:
        model = create_model(model_selected, input_size)
        if not model:
            raise ValueError("모델 생성이 실패했습니다.")
    except Exception as e:
        return jsonify({"status": "error", "message": f"모델 생성 중 오류 발생: {str(e)}"}), 500

    # 모델 저장 경로 설정
    save_dir = os.path.join(MODEL_FOLDER, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 모델 정보 초기화
    model_info = {
        'model_type': model_type,
        'model_name': model_name,
        'model_selected': model_selected,
        'parameters': {},  # 추후에 파라미터 추가 가능
        'framework': 'pytorch' if isinstance(model, torch.nn.Module) else 'sklearn',
        'model_path': '',  # 실제 모델 파일 경로
        'input_size': input_size,
        'train_loss': 10e8,  # 초기화된 학습 손실 값
        'val_loss': 10e8     # 초기화된 검증 손실 값
    }

    # 모델 저장
    try:
        if model_info['framework'] == 'pytorch':
            model_path = os.path.join(save_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), model_path)
        elif model_info['framework'] == 'sklearn':
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
        else:
            raise ValueError("지원되지 않는 프레임워크입니다.")
        model_info['model_path'] = model_path
    except Exception as e:
        return jsonify({"status": "error", "message": f"모델 저장 중 오류 발생: {str(e)}"}), 500

    # 모델 정보 JSON으로 저장
    try:
        metadata_path = os.path.join(save_dir, f"{model_name}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return jsonify({"status": "error", "message": f"메타데이터 저장 중 오류 발생: {str(e)}"}), 500

    return jsonify({"status": "success", "message": f"모델 {model_name}이(가) 저장되었습니다.", "model_path": model_info['model_path']}), 200



@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    data = request.json
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({'status': 'error', 'message': '모델 이름이 필요합니다.'}), 400

    # 모델 디렉토리 경로
    model_dir = os.path.join(MODEL_FOLDER, model_name)

    # 디렉토리가 존재하는지 확인
    if not os.path.exists(model_dir):
        return jsonify({'status': 'error', 'message': f'모델 {model_name}을(를) 찾을 수 없습니다.'}), 404

    if not os.path.isdir(model_dir):
        return jsonify({'status': 'error', 'message': f'{model_name}은(는) 디렉토리가 아닙니다.'}), 400

    try:
        # 디렉토리 내 파일과 디렉토리 삭제
        for root, dirs, files in os.walk(model_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생: {file_path}, {str(e)}")
                    return jsonify({'status': 'error', 'message': f"파일 삭제 중 오류 발생: {file_path}"}), 500
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    os.rmdir(dir_path)
                except Exception as e:
                    print(f"디렉토리 삭제 중 오류 발생: {dir_path}, {str(e)}")
                    return jsonify({'status': 'error', 'message': f"디렉토리 삭제 중 오류 발생: {dir_path}"}), 500

        # 최종 디렉토리 삭제
        os.rmdir(model_dir)

        return jsonify({'status': 'success', 'message': f'모델 {model_name}이(가) 삭제되었습니다.'}), 200

    except Exception as e:
        print(f"모델 삭제 중 오류 발생: {str(e)}")
        return jsonify({'status': 'error', 'message': f'모델 삭제 중 오류가 발생했습니다: {str(e)}'}), 500


@app.route('/api/get_models', methods=['GET'])
def get_models():
    models = []
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER, exist_ok=True)

    for model_name in os.listdir(MODEL_FOLDER):
        model_dir = os.path.join(MODEL_FOLDER, model_name)
        metadata_path = os.path.join(model_dir, f"{model_name}.json")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # 필요한 키만 추출하여 클라이언트에 반환
                    models.append({
                        'model_name': metadata.get('model_name'),
                        'framework': metadata.get('framework'),
                        'model_selected': metadata.get('model_selected'),
                        'input_size': metadata.get('input_size'),
                    })
            except Exception as e:
                print(f"모델 메타데이터 로드 오류: {str(e)}")
                continue

    return jsonify({'status': 'success', 'models': models})

'''---------------------------------------------모델학습----------------------------------------------'''
@app.route('/api/train_model', methods=['POST'])
def train_model():
    data = request.json
    csv_filename = data.get('csv_filename')
    model_name = data.get('model_name')
    target_column = data.get('target_column')
    val_ratio = float(data.get('val_ratio', 0.2))
    train_loss = 99999999
    val_loss = 99999999
    # 필수 데이터 확인
    if not csv_filename or not model_name or not target_column:
        return jsonify({'status': 'error', 'message': '필요한 정보가 부족합니다.'}), 400

    # CSV 파일 로드
    csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': f"CSV 파일 '{csv_filename}'을 찾을 수 없습니다."}), 404

    try:
        df = pd.read_csv(csv_path)
        print(df.head())  # 데이터 확인
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"CSV 파일 읽기 오류: {str(e)}"}), 500

    # Target 컬럼 확인
    if target_column not in df.columns:
        return jsonify({'status': 'error', 'message': f"Target 컬럼 '{target_column}'이 CSV 파일에 존재하지 않습니다."}), 400

    # 모델 정보 로드
    model_dir = os.path.join(MODEL_FOLDER, model_name)
    metadata_path = os.path.join(model_dir, f"{model_name}.json")
    if not os.path.exists(metadata_path):
        return jsonify({'status': 'error', 'message': f"모델 메타데이터 '{metadata_path}'를 찾을 수 없습니다."}), 404

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"메타데이터 로드 오류: {str(e)}"}), 500

    model_path = model_info.get('model_path')
    framework = model_info.get('framework')

    # 모델 경로 및 프레임워크 확인
    if not model_path or not framework:
        return jsonify({'status': 'error', 'message': '모델 정보가 잘못되었습니다.'}), 500

    # 모델 로드
    try:
        if framework == 'sklearn':
            model = joblib.load(model_path)
        elif framework == 'pytorch':
            print('this is pytorch')
        else:
            return jsonify({'status': 'error', 'message': '지원되지 않는 모델 프레임워크입니다.'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"모델 로드 오류: {str(e)}"}), 500
    
    # # 데이터 전처리 및 학습
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 결측값 처리
    df = df.fillna(0)

    # 스케일링 적용
    scaler = MinMaxScaling(df,target_column=target_column)
    X, y = scaler.get_scaled_data()


    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"데이터 분할 오류: {str(e)}"}), 500

    # 모델 학습
    try:
        if framework == 'sklearn':
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            train_loss = mean_squared_error(y_train, train_predictions)

            if X_val is not None:
                val_predictions = model.predict(X_val)
                val_loss = mean_squared_error(y_val, val_predictions)
            else:
                val_loss = None

            # 모델 저장
            joblib.dump(model, model_path)
        else:
            # 모델 학습
            # PyTorch 모델 로드
            input_size = model_info.get('input_size')
            model_selected = model_info.get('model_selected')
            
                # JSON 파일에서 모델 설정 값 가져오기
            model_selected = model_info.get('model_selected')
            input_size = model_info.get('input_size')
            parameters = model_info.get('parameters', {})  # 모델 파라미터 정보
            # MLP 모델 초기화
            if model_selected.startswith('MLP'):
                hidden_size = parameters.get('hidden_size', 32)  # 기본값: 32
                n_layers = parameters.get('n_layers', 1)        # 기본값: 1
                output_size = parameters.get('output_size', 1)  # 기본값: 1

                # MLP 클래스 사용하여 모델 생성
                model = MLP(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers, output_size=output_size)

            # 데이터셋 및 데이터로더 생성
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            if X_val is not None:
                X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            else:
                val_loader = None

            # 손실 함수 및 옵티마이저 정의
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 얼리 스탑핑 적용 (early_stopping.py에서 EarlyStopping 클래스 가져오기)
            from early_stopping import EarlyStopping  # 별도의 파일에 구현된 얼리 스탑핑 클래스

            early_stopping = EarlyStopping(patience=10, verbose=False, path=os.path.join(model_dir, 'checkpoint.pt'))

            num_epochs = 500
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                train_loss = train_loss / len(train_loader.dataset)

                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item() * inputs.size(0)
                    val_loss = val_loss / len(val_loader.dataset)

                    # 얼리 스탑핑 체크
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print(f"Epoch {epoch}: Early stopping")
                        break
                else:
                    val_loss = None

            # 최적의 모델 로드
            if val_loader is not None and os.path.exists(os.path.join(model_dir, 'checkpoint.pt')):
                model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoint.pt')))

            # 학습된 모델 저장
            torch.save(model.state_dict(), model_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"모델 학습 중 오류 발생: {str(e)}"}), 500
    # Denormalize된 값 계산
    denormalized_train_loss = scaler.denormalize([train_loss], [target_column])[0]
    denormalized_val_loss = scaler.denormalize([val_loss], [target_column])[0] if val_loss else None

    # 결과 반환
    result = {
        'model_name': model_name,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'denormalized_train_loss': denormalized_train_loss,
        'denormalized_val_loss': denormalized_val_loss
    }

    return jsonify({'status': 'success', 'message': f'모델 {model_name} 학습 완료', 'result': result})


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


'''--------------------------------------------------input_prediction----------------------------------------------------------------------------'''
@app.route('/api/submit_prediction', methods=['POST'])
def submit_prediction():
    try:
        # JSON 데이터 파싱
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '데이터가 전송되지 않았습니다.'}), 400

        # 필수 필드 확인
        required_fields = ['filename', 'desire', 'option', 'modeling_type', 'strategy', 'starting_points', 'units', 'models']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'{field} 필드가 필요합니다.'}), 400

        filename = data['filename']
        desire = data['desire']
        option = data['option']
        modeling_type = data['modeling_type']
        strategy = data['strategy']
        starting_points = data['starting_points']  # 딕셔너리
        units = data['units']                      # 딕셔너리
        models = data['models']                    # 리스트

        # 추가 정보 (옵션 및 전략에 따라)
        tolerance = data.get('tolerance')
        beam_width = data.get('beam_width')
        num_candidates = data.get('num_candidates')
        escape = data.get('escape')
        top_k = data.get('top_k')
        partial_keep = data.get('partial_keep')
        index = data.get('index')
        up = data.get('up')

        # CSV 파일 로드
        csv_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(csv_path):
            return jsonify({'status': 'error', 'message': f'파일 {filename}을 찾을 수 없습니다.'}), 404

        df = pd.read_csv(csv_path)

        # 시작점 데이터를 모델 입력 형식에 맞게 변환
        # 단위에 따른 데이터 처리는 여기서 하지 않습니다. 해당 구현은 다른 곳에서 정의될 예정입니다.
        input_data = []
        for col in df.columns:
            if col in starting_points:
                value = float(starting_points[col])
                input_data.append(value)
            elif col.lower() == 'target':
                continue  # Target 컬럼은 입력 데이터에서 제외
            else:
                # 시작점에 없는 컬럼은 평균값으로 대체하거나 다른 처리 필요
                value = df[col].mean()
                input_data.append(value)

        # 모델 로드 및 예측 수행
        predictions = {}
        for model_name in models:
            model_dir = os.path.join(MODEL_FOLDER, model_name)
            metadata_path = os.path.join(model_dir, f"{model_name}.json")

            if not os.path.exists(metadata_path):
                return jsonify({'status': 'error', 'message': f"모델 메타데이터 '{metadata_path}'를 찾을 수 없습니다."}), 404

            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)

            model_path = model_info.get('model_path')
            framework = model_info.get('framework')

            # 모델 로드
            if framework == 'sklearn':
                model = joblib.load(model_path)
                # 예측 수행
                prediction = model.predict([input_data])[0]
            elif framework == 'pytorch':
                # 기존 코드에서 사용한 방법으로 PyTorch 모델 로드
                input_size = model_info.get('input_size')
                model_selected = model_info.get('model_selected')
                parameters = model_info.get('parameters', {})

                # MLP 모델 초기화 (기존 코드의 방식 사용)
                if model_selected.startswith('MLP'):
                    hidden_size = parameters.get('hidden_size', 32)
                    n_layers = parameters.get('n_layers', 1)
                    output_size = parameters.get('output_size', 1)

                    model = MLP(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers, output_size=output_size)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    with torch.no_grad():
                        input_tensor = torch.tensor([input_data], dtype=torch.float32)
                        output = model(input_tensor)
                        prediction = output.item()
                else:
                    return jsonify({'status': 'error', 'message': f"지원되지 않는 PyTorch 모델 유형입니다: {model_selected}"}), 400
            else:
                return jsonify({'status': 'error', 'message': f"지원되지 않는 모델 프레임워크입니다: {framework}"}), 400

            predictions[model_name] = prediction

        # 모델링 타입에 따라 결과 처리
        if modeling_type == 'Single':
            # 단일 모델의 예측 결과 반환
            result = predictions[models[0]]  # 첫 번째 모델의 결과
        elif modeling_type == 'Averaging':
            # 예측 결과들의 평균 반환
            result = sum(predictions.values()) / len(predictions)
        elif modeling_type == 'Ensemble':
            # 앙상블 방법 적용 (예: 다수결, 가중 평균 등)
            # 여기서는 단순 평균을 사용
            result = sum(predictions.values()) / len(predictions)
        else:
            return jsonify({'status': 'error', 'message': f'알 수 없는 모델링 타입: {modeling_type}'}), 400

        # 옵션 및 전략에 따른 추가 처리
        # 해당 부분은 구현 예정이므로 현재는 패스

        # 최종 결과 반환
        return jsonify({'status': 'success', 'result': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'예측 중 오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)