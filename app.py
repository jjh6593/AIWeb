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
import time
from sklearn.metrics import mean_squared_error
from train import train_pytorch_model, train_sklearn_model
import logging
from traking import run
import shutil
# 디버깅 로깅 설정
logging.basicConfig(level=logging.DEBUG)
# CORS 설정을 위해 필요하다면 다음 코드를 추가하세요.
from flask_cors import CORS
import numpy as np
app = Flask(__name__)

CORS(app)
# 업로드 및 모델 저장 디렉토리 설정


UPLOAD_FOLDER = 'uploads'
OUTPUTS_FOLDER = 'outputs'
MODEL_FOLDER = 'models'

# Flask 설정에 디렉토리 경로 추가
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUTS_FOLDER'] = OUTPUTS_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)



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
    print(f"Received data: {data}")  # 디버깅용
    exclude_columns = data.get('exclude_columns', [])
    new_filename = data.get('new_filename', '')
    filename = data.get('filename', '')  # JSON에서 filename 가져오기

    if not new_filename:
        return jsonify({'status': 'error', 'message': '새 파일 이름이 필요합니다.'}), 400

    if not filename:
        return jsonify({'status': 'error', 'message': '원본 파일명이 필요합니다.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # current_app 대신 app 사용
    print(f"File path: {file_path}")  # 디버깅용
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': '원본 파일을 찾을 수 없습니다.'}), 404

    # CSV 파일에서 제외된 컬럼을 제거하고 저장
    df = pd.read_csv(file_path)
    filtered_df = df.drop(columns=exclude_columns, errors='ignore')
    new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    print(f"Saved filtered file to: {new_file_path}")
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
                        'train_loss': metadata.get('train_loss'),
                        'val_loss': metadata.get('val_loss')
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
    # JSON 파일 업데이트
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # train_loss, val_loss 값을 업데이트
        metadata['train_loss'] = train_loss
        metadata['val_loss'] = val_loss if val_loss is not None else "N/A"

        # 업데이트된 메타데이터 저장
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f"JSON 파일 업데이트 오류: {str(e)}"}), 500

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
def debug_training_results():
    results = []

    if not os.path.exists(OUTPUTS_FOLDER):
        print(f"'{OUTPUTS_FOLDER}' 폴더가 존재하지 않습니다.")
        return {'status': 'error', 'message': f"'{OUTPUTS_FOLDER}' 폴더가 존재하지 않습니다."}

    print(f"'{OUTPUTS_FOLDER}' 폴더를 탐색합니다...")
    for folder_name in os.listdir(OUTPUTS_FOLDER):
        folder_path = os.path.join(OUTPUTS_FOLDER, folder_name)
        output_file_path = os.path.join(folder_path, f"{folder_name}_output.json")

        if os.path.exists(output_file_path):
            print(f"파일 발견: {output_file_path}")
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                    best_config = output_data.get('best_config')
                    best_pred = output_data.get('best_pred')

                    if best_config is not None and best_pred is not None:
                        print(f"'{folder_name}'에서 읽은 데이터:")
                        print(json.dumps({
                            'folder_name': folder_name,
                            'best_config': best_config,
                            'best_pred': best_pred
                        }, indent=4))

                        results.append({
                            'folder_name': folder_name,
                            'best_config': best_config,
                            'best_pred': best_pred
                        })
                    else:
                        print(f"'{output_file_path}'에서 'best_config' 또는 'best_pred'가 없습니다.")
            except Exception as e:
                print(f"파일 읽기 실패: {output_file_path}, 오류: {e}")
        else:
            print(f"'{folder_path}' 폴더에 '{folder_name}_output.json' 파일이 없습니다.")

    if not results:
        print("결과가 없습니다.")
    else:
        print("최종 결과:")
        print(json.dumps({'status': 'success', 'results': results}, indent=4))

    return {'status': 'success', 'results': results}
'''--------------------------------------------------input_prediction----------------------------------------------------------------------------'''
def process_models(models_input):
    # 매핑 사전 정의
    mapping = {
        'MLP_1': 'MLP()',
        'MLP_2': 'MLP(n_layers = 2)',
        'MLP_3': 'MLP(n_layers = 3)',
        'LinearRegressor': 'ML_LinearRegressor()',
        'Ridge': 'ML_Ridge()',
        'Lasso': 'ML_Lasso()',
        'ElasticNet': 'ML_ElasticNet()',
        'DecisionTreeRegressor': 'ML_DecisionTreeRegressor()',
        'RandomForestRegressor': 'ML_RandomForestRegressor()',
        'GradientBoostingRegressor': 'ML_GradientBoostingRegressor()',
        'SVR': 'ML_SVR()',
        'KNeighborsRegressor': 'ML_KNeighborsRegressor()',
        'HuberRegressor': 'ML_HuberRegressor()',
        'GaussianProcessRegressor': 'ML_GaussianProcessRegressor()',
        'XGBoost': 'ML_XGBoost()'
    }

    # 매핑된 모델 리스트 초기화
    mapped_models = []

    # 모델 처리 시작
    for model_name in models_input:
        # 모델 저장 경로 설정
        save_dir = os.path.join(MODEL_FOLDER, model_name)
        os.makedirs(save_dir, exist_ok=True)

        # 메타데이터 파일 경로
        metadata_path = os.path.join(save_dir, f"{model_name}.json")
        if not os.path.exists(metadata_path):
            print(f"Metadata for model {model_name} not found. Skipping...")
            continue

        try:
            # 메타데이터 로드
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # model_selected 값 가져오기
            model_selected = metadata.get('model_selected')
            if model_selected and model_selected in mapping:
                # 매핑 결과 추가
                mapped_models.append(mapping[model_selected])
            else:
                print(f"No valid mapping for model_selected: {model_selected}. Skipping...")
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

    return mapped_models
def convert_to_serializable(obj):
    if obj is None:
        return None  # None은 JSON에서 null로 직렬화됨
    if isinstance(obj, (np.float32, np.float64)):  # numpy float 처리
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):  # numpy int 처리
        return int(obj)
    if isinstance(obj, list):  # 리스트 처리
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, dict):  # 딕셔너리 처리
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj  # 변환이 필요 없는 경우 그대로 반환

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"An error occurred: {e}")
    return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/submit_prediction', methods=['POST'])
def submit_prediction():
    try:
        # 요청에서 JSON 데이터 가져오기
        data = request.get_json()
        

        # 터미널에 받은 데이터 출력
        logging.debug("Received data:")
        logging.debug(json.dumps(data, indent=4, ensure_ascii=False))

        # 필수 필드 확인
        required_fields = ['filename', 'desire', 'save_name', 'option', 'modeling_type', 'strategy', 'starting_points', 'units', 'models']
        for field in required_fields:
            if field not in data:
                logging.error(f"Missing field: {field}")
                return jsonify({'status': 'error', 'message': f'필수 필드가 누락되었습니다: {field}'}), 400

        # 'option' 필드 검증
        if data['option'] not in ['local', 'global']:
            logging.error("Invalid option value")
            return jsonify({'status': 'error', 'message': '옵션 값이 올바르지 않습니다. "local" 또는 "global"이어야 합니다.'}), 400

        # 'global' 옵션일 경우 추가 필드 확인
        if data['option'] == 'global':
            if 'tolerance' not in data:
                logging.error("Missing tolerance field for global option")
                return jsonify({'status': 'error', 'message': 'Global 옵션에서는 tolerance 필드가 필요합니다.'}), 400

            if data['strategy'] == 'Beam' and 'beam_width' not in data:
                logging.error("Missing beam_width field for Beam strategy")
                return jsonify({'status': 'error', 'message': 'Beam 전략에서는 beam_width 필드가 필요합니다.'}), 400

        required_keys = ['filename', 'desire', 'save_name', 'option', 'modeling_type', 'strategy']
        for key in required_keys:
            if key not in data or data[key] is None:
                raise ValueError(f"Missing or invalid value for key: {key}")

        # 데이터 타입 검증
        try:
            data['desire'] = float(data['desire'])
        except ValueError:
            logging.error("Invalid desire value: must be a float")
            return jsonify({'status': 'error', 'message': 'desire 값은 실수여야 합니다.'}), 400

        # 데이터 파일 로드
        data_file_path = os.path.join('uploads', data['filename'])
        if not os.path.exists(data_file_path):
            logging.error(f"Data file not found: {data['filename']}")
            return jsonify({'status': 'error', 'message': f'Data file not found: {data["filename"]}'}), 400

        df = pd.read_csv(data_file_path).drop_duplicates()
        model_list = process_models(data['models'])
        # model_list = ['MLP()', "ML_XGBoost()"]
        #starting_point = data['starting_points']
        models = None
        mode = data['option']
        print(mode)
        desired = data['desire']
        print(desired)
        modeling = data['modeling_type']
        print(modeling)
        strategy = data['strategy']
        print(strategy)
        tolerance = data.get('tolerance', None)
        beam_width = data.get('beam_width', None)
        num_candidates = data.get('num_candidates', 5)
        escape = data.get('escape', True)
        top_k = data.get('top_k', 2)
        index = data.get('index', 0)
        up = data.get('up', 0)
        alternative = data.get('alternative', 'keep_move')
        

        # 파일 이름 생성
        print(data['save_name'])
        save_name = data['save_name']
        outputs_dir = 'outputs'
        os.makedirs(outputs_dir, exist_ok=True)

        # # 파일 경로 설정
        # input_file_path = os.path.join(outputs_dir, f'{save_name}_input.json')
        # output_file_path = os.path.join(outputs_dir, f'{save_name}_output.json')
        # 파일 이름 및 경로 설정
        save_name = data['save_name']
        outputs_dir = 'outputs'
        os.makedirs(outputs_dir, exist_ok=True)

        # 서브 폴더 설정
        subfolder_path = os.path.join(outputs_dir, save_name)

        # 동일한 폴더가 있으면 그대로 사용
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # 파일 경로 설정 (폴더명 기반 파일 이름 생성)
        input_file_path = os.path.join(subfolder_path, f'{save_name}_input.json')
        output_file_path = os.path.join(subfolder_path, f'{save_name}_output.json')
        # # 동일한 파일명이 있을 경우 처리 (숫자 증가)
        # counter = 1

        # while os.path.exists(input_file_path) or os.path.exists(output_file_path):
        #     input_file_path = os.path.join(outputs_dir, f'{save_name}_{counter}_input.json')
        #     output_file_path = os.path.join(outputs_dir, f'{save_name}_{counter}_output.json')
        #     counter += 1

        models, training_losses, configurations, predictions, best_config, best_pred = run(
            data=df,  # DataFrame
            models=models,
            model_list=model_list,
            desired=float(desired),  # 실수형으로 변환
            starting_point=[float(value) for value in data['starting_points'].values()],  # 숫자 리스트로 변환
            mode=mode,
            modeling=modeling.lower(),
            strategy=strategy.lower(),
            tolerance=float(tolerance) if tolerance is not None else None,  # 실수형 변환
            beam_width=int(beam_width) if beam_width is not None else None,
            num_cadidates=int(num_candidates),
            escape=bool(escape),
            top_k=int(top_k),
            index=int(index),
            up=bool(up),
            alternative=alternative
        )

        output_data = {
            'training_losses': convert_to_serializable(training_losses),
            'configurations': convert_to_serializable(configurations),
            'predictions': convert_to_serializable(predictions),
            'best_config': convert_to_serializable(best_config),
            'best_pred': convert_to_serializable(best_pred),
        }

        # 데이터 저장
        with open(input_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        # 성공 응답
        return jsonify({'status': 'success', 'data': output_data}), 200

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

'''---------------------------------------------학습 결과----------------------------------------------'''
# 학습 결과 가져오기
@app.route('/api/get_training_results', methods=['GET'])
def get_training_results():
    results = []

    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    for folder_name in os.listdir(OUTPUTS_FOLDER):
        folder_path = os.path.join(OUTPUTS_FOLDER, folder_name)
        
        output_file_path = os.path.join(folder_path, f"{folder_name}_output.json")
        
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                    results.append({
                        'folder_name': str(folder_name),
                        'predictions' : output_data.get('predictions'),
                        'configurations': output_data.get('configurations'),
                        'best_config': output_data.get('best_config'),
                        'best_pred': output_data.get('best_pred'),
                    })
            except Exception as e:
                print(f"Error reading {output_file_path}: {e}")
                continue

    return jsonify({'status': 'success', 'results': results})


# 결과 삭제 엔드포인트
@app.route('/api/delete_result', methods=['POST'])
def delete_result():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'status': 'error', 'message': 'Filename is required.'}), 400

    outputs_dir = 'outputs'
    folder_path = os.path.join(outputs_dir, filename)
    print(filename)
    try:
        # Ensure the folder_path is within the outputs_dir to prevent directory traversal attacks
        outputs_dir_abs = os.path.abspath(outputs_dir)
        folder_path_abs = os.path.abspath(folder_path)
        if not folder_path_abs.startswith(outputs_dir_abs):
            return jsonify({'status': 'error', 'message': 'Invalid filename.'}), 400

        if os.path.exists(folder_path):
            # Delete the directory and all its contents
            shutil.rmtree(folder_path)
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Folder does not exist.'}), 404

    except Exception as e:
        logging.error(f"Error deleting folder {folder_path}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)