from flask import Flask, request, jsonify, send_from_directory, render_template, make_response
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os, datetime
import random
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
import mimetypes
import pickle
import json
from model import create_model, _train_nn
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
from traking import parameter_prediction
import shutil
from sklearn.model_selection import KFold
# 디버깅 로깅 설정
logging.basicConfig(level=logging.DEBUG)
# CORS 설정을 위해 필요하다면 다음 코드를 추가하세요.
from flask_cors import CORS
import numpy as np
app = Flask(__name__)

# CORS(app)
# 특정 Origin 허용
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173","http://localhost:5173/", "http://127.0.0.1:5173/"]}})

# 업로드 및 모델 저장 디렉토리 설정
mimetypes.init()
mimetypes.add_type('application/javascript', '.js', strict=True)

UPLOAD_FOLDER = 'uploads'
OUTPUTS_FOLDER = 'outputs'
MODEL_FOLDER = 'models'
METADATA_FOLDER = 'metadata'  # 메타데이터 저장 폴더

# Flask 설정에 디렉토리 경로 추가
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUTS_FOLDER'] = OUTPUTS_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)

# 라우트 설정

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static_file(path):
    return send_from_directory('static', path)

# API 엔드포인트
@app.before_request
def handle_options_request():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin"))
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Max-Age", "3600")
        return response


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

# -------------------------
# 새롭게 추가된 메타데이터 저장(생성/수정) API
# -------------------------
@app.route('/api/save_csv_metadata', methods=['POST'])
def save_csv_metadata():
    """
    메타데이터를 새로 생성하거나, 기존 메타데이터를 갱신(덮어쓰기)하기 위한 엔드포인트.
    - 요청 예:
        {
          "filename": "example.csv",
          "metadata": [
            {
              "column": "Temperature",
              "unit": "C",
              "min": 0,
              "max": 100
            },
            {
              "column": "Pressure",
              "unit": "bar",
              "min": 1,
              "max": 50
            }
          ]
        }
    - 응답 예:
        {
          "status": "success",
          "message": "Metadata saved/updated successfully.",
          "saved_metadata": [...]  // 저장된 메타데이터 그대로
        }
    """
    data = request.json
    print(data)
    filename = data.get('filename')
    metadata = data.get('metadata', [])

    if not filename:
        return jsonify({'status': 'error', 'message': 'filename 파라미터가 필요합니다.'}), 400

    if not metadata:
        return jsonify({'status': 'error', 'message': 'metadata가 비어 있습니다.'}), 400

    # 실제 CSV가 존재하는지 체크 (선택적으로)
    csv_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': '해당 CSV 파일이 존재하지 않습니다.'}), 404
    # (1) unit, min, max를 float으로 변환
    
    for item in metadata:
        # get()에서 디폴트값 0.0 (또는 None)으로 지정해도 됨
        item['unit'] = float(item.get('unit', 0.0))
        item['min'] = float(item.get('min', 0.0))
        item['max'] = float(item.get('max', 0.0))
        item['round'] = int(item.get('round',0))
        # data_type 문자열을 실제 Python 타입으로 변환
        data_type_str = item.get('data_type', 'float').lower()
        # 그냥 문자열 그대로 저장하기
        item['data_type'] = data_type_str
        
    # 메타데이터를 저장할 경로
    metadata_path = os.path.join(METADATA_FOLDER, f"{filename}_metadata.json")
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return jsonify({
            'status': 'success',
            'message': 'Metadata saved/updated successfully.',
            'saved_metadata': metadata
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


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

# # 6. 모델 생성
@app.route('/api/save_model', methods=['POST'])
def save_model():
    
    data = request.json
    if not data:
        return jsonify({'status': 'error', 'message': 'JSON 데이터가 제공되지 않았습니다.'}), 400

    # 디버깅용: 요청 데이터 출력
    print(f"Received data: {data}")
    model_type = data.get('model_type')
    model_name = data.get('model_name')
    model_selected = data.get('model_selected')
    # input_size = data.get('input_size', None)
    input_size = data.get('hyperparameters', {}).get('input_size', None)

    csv_filename = data.get('csv_filename')
    hyperparams = data.get('hyperparameters', {})
    target_column = "Target"
    epochs = hyperparams.get('epoch', 100) # 디폴트 0.001
    save_dir = os.path.join(MODEL_FOLDER, model_name)
    os.makedirs(save_dir, exist_ok=True)

    val_ratio = float(data.get('val_ratio', 0.2))
    # 필수 필드 누락 확인
    if not model_type or not model_name or not model_selected or not csv_filename or not target_column:
        missing_fields = [field for field in ['model_type', 'model_name', 'model_selected', 'csv_filename', 'target_column'] if not data.get(field)]
        print(f"Missing fields: {missing_fields}")
        return jsonify({'status': 'error', 'message': f'필수 정보가 누락되었습니다: {missing_fields}'}), 400
    
    # 필수 데이터 확인
    if not model_type or not model_name or not model_selected or not csv_filename or not target_column:
        return jsonify({'status': 'error', 'message': '필요한 정보가 부족합니다.'}), 400
    
    # CSV 파일 로드
    csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': f"CSV 파일 '{csv_filename}'을 찾을 수 없습니다."}), 404

    try:
        df = pd.read_csv(csv_path)
        #print(f"CSV 데이터 미리보기: {df.head()}")  # 디버깅용
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"CSV 파일 읽기 오류: {str(e)}"}), 500

    # Target 컬럼 확인
    if target_column not in df.columns:
        return jsonify({'status': 'error', 'message': f"Target 컬럼 '{target_column}'이 CSV 파일에 존재하지 않습니다."}), 400
    

    '''-----------------------------제약 조건에 따라서 모델 생성 ------------------------------'''
    # (2) 메타데이터 로드 & constraints 생성
    metadata_path = os.path.join(METADATA_FOLDER, f"{csv_filename}_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)


    # -----------------------------------------------------------------------------
    # ### (A) 각 열의 min == max인 경우, 해당 열을 erase_cols에 추가 후 학습에서 제외
    # -----------------------------------------------------------------------------
    erase_cols = []
    for col in df.columns:
        if col == target_column:
            continue  # 타겟열은 무조건 사용한다는 가정 (원하면 이 조건 제거)
        if df[col].min() == df[col].max():
            erase_cols.append(col)
    print(erase_cols)
    print(df.shape)

    if erase_cols:
        print(f"[DEBUG] 변동이 없는(최소=최대) 열들: {erase_cols}. 학습에서 제외합니다.")
        df.drop(columns=erase_cols, inplace=True)

    # (B) 메타데이터 필터링
    # 기존 metadata_list에서, erase_cols에 해당하는 column의 item은 제외
    filtered_metadata = [
        item for item in metadata_list 
        if item['column'] not in erase_cols
    ]
    # 모델 생성
    if input_size is not None:
        # user가 준 input_size에서 erase_cols 만큼 빼준다
        new_input_size = input_size - len(erase_cols)
        if new_input_size < 1:
            raise ValueError(f"유효한 input_size가 아닙니다 (지워진 컬럼이 너무 많음).")
        hyperparams["input_size"] = new_input_size
        input_size = new_input_size
    
    try:
        # GradientBoostingRegressor 또는 RandomForestRegressor일 경우 외부에서 생성
        if model_selected == 'RandomForestRegressor':
            model = RandomForestRegressor(**hyperparams)
            print(f"모델 생성 성공 (외부): {model}")
        elif model_selected == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(**hyperparams)
            print(f"모델 생성 성공 (외부): {model}")
        else:
            # 그 외 모델은 기존 로직 유지
            model = create_model(model_selected, input_size, hyperparams=hyperparams)
            print(f"모델 생성 성공: {model}")  # 디버깅용
            if not model:
                raise ValueError("모델 생성 실패")

    except Exception as e:
        return jsonify({"status": "error", "message": f"모델 생성 중 오류 발생: {str(e)}"}), 500

    
        
    # constraints 구성
    constraints = {}
    for item in  filtered_metadata:
        col_name = item['column']       # "att1" 등
        unit_val = float(item['unit'])  # 5, 0.5 등
        min_val  = float(item['min'])   # 예: 0
        max_val  = float(item['max'])   # 예: 100
        # 임의로 결정한 타입, 반올림 자릿수
        # 실제는 item에 "type" / "round_digits"가 있으면 거기서 그대로 읽어도 됨
        dtype_val = float if not float(unit_val).is_integer() else int
        round_digits = 2 if dtype_val == float else 0

        constraints[col_name] = [unit_val, min_val, max_val, dtype_val, round_digits]

    
    # ---------------------------
    # (3) 결측치 처리, X / y 분리
    # ---------------------------
    df = df.fillna(0)
    X_df = df.drop(columns=[target_column])
    y_df = df[[target_column]]  # 2D
    print(X_df.shape)
    scaler_X = MinMaxScaling(data=X_df, constraints=constraints)  # dtype = torch.float32(기본값)
    X_scaled = scaler_X.data.detach().numpy()  # (N, D) numpy
    scaler_y = MinMaxScaler()  
    y_scaled = scaler_y.fit_transform(y_df).flatten()
    
     # ---------------------------
    # (4) 데이터 분할
    # ---------------------------
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=val_ratio, random_state=42)

    # feature, target을 np.array 또는 torch.tensor 형태로 변환 (기존 로직 그대로)
    X_train_np = X_train
    y_train_np = y_train.reshape(-1, 1)
    X_val_np = X_val
    y_val_np = y_val.reshape(-1, 1)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    
    # 모델 학습
    try:
        #print("모델 학습 시작...")
        if isinstance(model, torch.nn.Module):
            training_losses = _train_nn(model, X_train_np, y_train_np, epochs)
        else:
            model.fit(X_train_np, y_train_np)  # 여기서 에러 발생 가능성 확인
        #print("모델 학습 완료.")
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        return jsonify({"status": "error", "message": f"모델 학습 중 오류 발생: {str(e)}"}), 500
    
    # 폴더 생성
    
    model_path = os.path.join(save_dir, f"{model_name}.pt" if isinstance(model, torch.nn.Module) else f"{model_name}.pkl")
    modeldata_path = os.path.join(save_dir, f"{model_name}.json")
    creation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

   # 모델 저장
    try:
        print("모델 저장 시도...")
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_path)
        else:
            joblib.dump(model, model_path)  # 여기서 에러 발생 시 추적
        print("모델 저장 완료.")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {str(e)}")
        return jsonify({"status": "error", "message": f"모델 저장 중 오류 발생: {str(e)}"}), 500

    # 모델 불러오기 검증
    try:
        if isinstance(model, torch.nn.Module):
            # PyTorch 모델
            loaded_model = create_model(model_selected, input_size, hyperparams=hyperparams)  # 모델 구조 생성
            loaded_model.load_state_dict(torch.load(model_path))
            print(f"PyTorch 모델 로드 성공: {loaded_model}")
        else:
            # scikit-learn 모델
            loaded_model = joblib.load(model_path)
            print(f"scikit-learn 모델 로드 성공: {loaded_model}")
    except Exception as e:
        print(f"모델 불러오기 중 오류 발생: {str(e)}")
    # Metrics 계산
    def model_predict(model, X):
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32)
                # 입력 데이터가 1D인 경우, 2D로 변환
                if len(X_t.shape) == 1:
                    X_t = X_t.view(-1, 1)
                preds = model(X_t).detach().numpy().flatten()
            return preds
        else:
            return model.predict(X)

    train_predictions = model_predict(model, X_train_np)
    val_predictions = model_predict(model, X_val_np) if len(X_val_np) > 0 else None
    train_pred_inv = scaler_y.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
    val_pred_inv = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten() if val_predictions is not None else None

    train_mse = mean_squared_error(y_train_np, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_rae = rae(y_train_np, train_predictions)

    if val_predictions is not None:
        val_mse = mean_squared_error(y_val_np, val_predictions)
        val_rmse = np.sqrt(val_mse)
        val_rae = rae(y_val_np, val_predictions)
    else:
        val_mse = val_rmse = val_rae = None

    # # 비정규화 (y 값만 역변환)
    y_train_inv = scaler_y.inverse_transform(y_train_np)
    y_val_inv = scaler_y.inverse_transform(y_val_np) if y_val_np is not None else None
    # y 자체도 역정규화
    # y_train_inv = scaler_y.denormalize(y_train_np).flatten()
    # y_val_inv = scaler_y.denormalize(y_val_np).flatten() if y_val_np is not None else None

    if model_type =='pytorch':
        train_pred_inv = scaler_y.inverse_transform(train_predictions.reshape(-1,1)).flatten()
        val_pred_inv = scaler_y.inverse_transform(val_predictions.reshape(-1,1)).flatten() if val_predictions is not None else None
        
    if y_train_inv is not None:
        train_mse_inv = mean_squared_error(y_train_inv, train_pred_inv)
        train_rmse_inv = np.sqrt(train_mse_inv)
        train_rae_inv = rae(y_train_inv, train_pred_inv)
    else:
        train_mse_inv = train_rmse_inv = train_rae_inv = None

    if val_pred_inv is not None and y_val_inv is not None:
        val_mse_inv = mean_squared_error(y_val_inv, val_pred_inv)
        val_rmse_inv = np.sqrt(val_mse_inv)
        val_rae_inv = rae(y_val_inv, val_pred_inv)
    else:
        val_mse_inv = val_rmse_inv = val_rae_inv = None

    # 메타데이터 저장
    model_info = {
        'model_type': model_type,
        'model_name': model_name,
        'model_selected': model_selected,
        'parameters': hyperparams,
        'framework': 'pytorch' if isinstance(model, torch.nn.Module) else 'sklearn',
        'model_path': model_path,
        'input_size': input_size,
        'train_loss': train_mse,
        'csv_filename': csv_filename,
        'val_loss': val_mse if val_mse is not None else "N/A",
        'creation_time': creation_time,
        'metrics': {
            'scaled': {
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_rae': train_rae,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_rae': val_rae
            },
            'original': {
                'train_mse': train_mse_inv,
                'train_rmse': train_rmse_inv,
                'train_rae': train_rae_inv,
                'val_mse': val_mse_inv,
                'val_rmse': val_rmse_inv,
                'val_rae': val_rae_inv
            }
        }
    }

    try:
        with open(modeldata_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"메타데이터 저장 중 오류 발생: {str(e)}"}), 500
    print(erase_cols)
    # 결과 반환
    result = {
        'status': 'success',
        'message': f'모델 {model_name} 생성 및 학습 완료',
        'model_name': model_name,
        'csv_filename': csv_filename,
        'creation_time': creation_time,
        'hyperparameters': hyperparams,
        'metrics': model_info['metrics'],
        'erase_col': erase_cols
    }

    return jsonify(result), 200

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

                    # 필요한 데이터 추출
                    model_info = {
                        'model_name': metadata.get('model_name'),
                        'framework': metadata.get('framework'),
                        'model_selected': metadata.get('model_selected'),
                        'input_size': metadata.get('input_size'),
                        'train_loss': metadata.get('train_loss'),
                        'val_loss': metadata.get('val_loss'),
                        'creation_time': metadata.get('creation_time'),  # 날짜 정보 추가
                        'parameters': metadata.get('parameters'),  # 파라미터 정보 추가
                        'csv_filename': metadata.get('csv_filename'),
                    }

                    # 스케일되지 않은 메트릭 추가
                    original_metrics = metadata.get('metrics', {}).get('original', {})
                    model_info.update({
                        'val_rae_original': original_metrics.get('val_rae'),
                        'val_rmse_original': original_metrics.get('val_rmse'),
                    })

                    models.append(model_info)

            except Exception as e:
                print(f"모델 메타데이터 로드 오류: {str(e)}")
                continue

    return jsonify({'status': 'success', 'models': models})


'''---------------------------------------------모델학습----------------------------------------------'''
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
        required_fields = ['filename', 'desire', 'save_name', 'option', 'modeling_type', 'strategy', 'starting_points', 'models']
        for field in required_fields:
            if field not in data:
                logging.error(f"Missing field: {field}")
                return jsonify({'status': 'error', 'message': f'필수 필드가 누락되었습니다: {field}'}), 400

        # filename으로 CSV 및 메타데이터를 찾음
        data_file_path = os.path.join('uploads', data['filename'])
        if not os.path.exists(data_file_path):
            logging.error(f"Data file not found: {data['filename']}")
            return jsonify({'status': 'error', 'message': f'Data file not found: {data["filename"]}'}), 400

        # (1) **메타데이터 로드**: filename 기반
        metadata_path = os.path.join(METADATA_FOLDER, f"{data['filename']}_metadata.json")
        if not os.path.exists(metadata_path):
            logging.error(f"Metadata file not found for: {data['filename']}")
            return jsonify({'status': 'error', 'message': f'Metadata file not found for: {data["filename"]}'}), 400

        with open(metadata_path, 'r', encoding='utf-8') as f:
            stored_metadata = json.load(f)
        units = [item['unit'] for item in stored_metadata]
        lower_bound = [item['min'] for item in stored_metadata]
        upper_bound = [item['max'] for item in stored_metadata]
        round_values = [item['round'] for item in stored_metadata]
        data_type = [item['data_type'] for item in stored_metadata]
        # CSV -> DataFrame 로드
        df = pd.read_csv(data_file_path).drop_duplicates()

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
        print(df.shape)
        # model_list = process_models(data['models'])
        
        models = None
        mode = data['option'].lower()
        print(mode)
        desired = int(data['desire'])
        print(desired)
        modeling = data['modeling_type'].lower()
        print(modeling)
        strategy = data['strategy'].lower()
        print(strategy)
        tolerance = data.get('tolerance', None)
        beam_width = data.get('beam_width', None)
        num_candidates = data.get('num_candidates', None)
        print(f"num_candidates = {num_candidates}")
        escape = data.get('escape', True)
        top_k = data.get('top_k', 2)
        index = data.get('index', 0)
        up = data.get('up', True)
        alternative = data.get('alternative', 'keep_move')
        # ============================
        # 기존 제약조건 가져오기
        # ============================
        metadata_path = os.path.join(METADATA_FOLDER, f"{data['filename']}_metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        # -----------------------------------------------------------------------------
        # ### (A) 각 열의 min == max인 경우, 해당 열을 erase_cols에 추가 후 학습에서 제외
        # -----------------------------------------------------------------------------
        erase_cols = []
        erase_indices = []  # 삭제할 컬럼들의 인덱스 저장
        feature_cols = [col for col in df.columns if col != 'Target']

        for idx, col in enumerate(feature_cols):
            if df[col].min() == df[col].max():
                erase_cols.append(col)
                erase_indices.append(idx)

        if erase_cols:
            print(f"[DEBUG] 변동이 없는(최소=최대) 열들: {erase_cols}. 학습에서 제외합니다.")
            df.drop(columns=erase_cols, inplace=True)

        # (B) 메타데이터 필터링
        # 기존 metadata_list에서, erase_cols에 해당하는 column의 item은 제외
        filtered_metadata = [
            item for item in metadata_list 
            if item['column'] not in erase_cols
        ]
        # (C) erase_indices를 사용해 리스트들 필터링
        def filter_by_indices(original_list, indices_to_remove):
            """indices_to_remove에 해당하는 원소를 original_list에서 제거"""
            return [val for idx, val in enumerate(original_list) if idx not in indices_to_remove]
                # data_type, decimal_place, starting_point 필터링
        
        
        # 변환할 딕셔너리 초기화
        units = []
        max_boundary = []
        min_boundary = []
        decimal_place = []
        data_type = []
        starting_point = []
        
        # 각 metadata 항목을 순회하며, column명을 key로 하여 value를 저장
        # filtered_metadata에서 배열 값 추출
        for item in filtered_metadata:
            units.append(item["unit"])
            max_boundary.append(item["max"])
            min_boundary.append(item["min"])
            decimal_place.append(item["round"])
            data_type.append(item["data_type"])
        
        data_type_mapping = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool
        }
        data_type = [data_type_mapping[dt] for dt in data_type]
        
        starting_point = list(data['starting_points'].values())
        print(starting_point)
        # starting_point = [150, 25, 40, 1, 120, 250, 10, 25, 25, 900, 0.25, 2, 100, 1800, 2000]
        starting_point = filter_by_indices(starting_point, erase_indices)

        # ============================
        # 여기서부터 모델 생성/파라미터 로드 로직
        # ============================
        MODEL_FOLDER = 'models'
        models_list = []

        for model_name in data['models']:
            model_dir = os.path.join(MODEL_FOLDER, model_name)
            model_metadata_path = os.path.join(model_dir, f"{model_name}.json")
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            
            if not os.path.exists(metadata_path):
                logging.error(f"No metadata found for model: {model_name}")
                continue

            try:
                # 메타데이터 로드
                with open(model_metadata_path, 'r', encoding='utf-8') as f:
                    model_metadata = json.load(f)

                framework = model_metadata.get('framework')
                model_selected = model_metadata.get('model_selected')
                hyperparams = model_metadata.get('parameters', {})
                input_size = model_metadata.get('input_size', None)
                model_path = ""
                print(model_selected)
                # framework에 따라 model_path 설정
                if framework == 'sklearn':
                    model_path = os.path.join(model_dir, f"{model_name}.pkl")
                elif framework == 'pytorch':
                    model_path = os.path.join(model_dir, f"{model_name}.pt")
                else:
                    logging.error(f"Unsupported framework '{framework}' for model: {model_name}")
                    continue
                if framework == 'sklearn':
                    # (1) pkl 파일이 존재하면 바로 불러오기
                    try:
                        with open(model_path, 'rb') as pf:
                            model = joblib.load(pf)
                        print(f"모델 로드 성공 (pkl): {model}")
                    except Exception as e:
                        logging.error(f"Error while loading model '{model_name}' from pkl: {e}")
                        continue

                elif framework == 'pytorch':
                    # PyTorch 모델 로드 로직 (기존 그대로 유지)
                    model = create_model(model_selected, input_size=input_size, hyperparams=hyperparams)
                    if not model:
                        raise ValueError(f"PyTorch 모델 생성 실패: {model_name}")
                    print(f"PyTorch 모델 생성 성공: {model}")

                    if os.path.isfile(model_path):
                        state_dict = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(state_dict)
                        print(f"PyTorch 모델 파라미터 로드 성공: {model}")

                else:
                    raise ValueError(f"지원되지 않는 프레임워크입니다: {framework}")

                # 완성된 모델 리스트에 추가
                models_list.append(model)

            except Exception as e:
                logging.error(f"Error while creating/loading model '{model_name}': {e}")
                continue

        print("모델 생성/로딩 완료:")
        # for idx, m in enumerate(models_list):
        #     print(f"{idx+1}. {m}")

        # 파일 이름 생성
        print(data['save_name'])
        save_name = data['save_name']
        outputs_dir = 'outputs'
        os.makedirs(outputs_dir, exist_ok=True)
        # 파일 이름 및 경로 설정
        save_name = data['save_name']
        outputs_dir = 'outputs'
        os.makedirs(outputs_dir, exist_ok=True)

        # 서브 폴더 설정
        subfolder_path = os.path.join(outputs_dir, save_name)

        # 동일한 폴더가 있으면 그대로 사용
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        
        
        # alternative = 'keep_move'
        models = data['models']
        print(f'models : {models}')
        print(f'model_list{models_list}')
        # max_boundary 및 min_boundary를 DataFrame에서 직접 추출
        # ---------------------------------------min,max 값일일히 설정하는거 방지-------------------------------------------------
        upper_bound = df.max().tolist()  # 각 컬럼의 최대값
        lower_bound = df.min().tolist()  # 각 컬럼의 최소값
        
        # desired = int(desired)
        # desired = 550
        # mode = 'global'
        # modeling = 'ensemble'
        # strategy = 'stochastic'
        tolerance = 1
        beam_width = 5
        num_candidates = 5
        escape = True
        # top_k = 2
        # index = 0
        # up = True
        alternative = 'keep_move'
        pred_all = None  # 미리 선언
        # 파일 경로 설정 (폴더명 기반 파일 이름 생성)
        input_file_path = os.path.join(subfolder_path, f'{save_name}_input.json')
        output_file_path = os.path.join(subfolder_path, f'{save_name}_output.json')

        configurations, predictions, best_config, best_pred, pred_all  = parameter_prediction(data = df, models = models_list,
                                                          desired = desired,
                                                          starting_point = starting_point, 
                                                          mode = mode, modeling = modeling,
                                                          strategy = strategy, tolerance = tolerance, 
                                                          beam_width = beam_width,
                                                          num_candidates = num_candidates, escape = escape, 
                                                          top_k = top_k, index = index,
                                                          up = up, alternative = alternative,
                                                          unit = units,
                                                          lower_bound = lower_bound, 
                                                          upper_bound = upper_bound, 
                                                          data_type = data_type, decimal_place = decimal_place)
        
        # 현재 시각(날짜) 정보
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # 입력 데이터에도 날짜정보 추가
        data['timestamp'] = timestamp
        # configurations, predictions 등을 전부 파이썬 기본 자료형으로 바꾸기
        configurations = convert_to_python_types(configurations)
        predictions = convert_to_python_types(predictions)
        best_config = convert_to_python_types(best_config)
        best_pred = convert_to_python_types(best_pred)
        pred_all = convert_to_python_types(pred_all)
        
        if pred_all is not None and pred_all:
            pred_all = convert_to_python_types(pred_all)

        output_data = {
            'mode': data['option'],
            'timestamp': timestamp,  # output에도 동일 날짜 정보
            'configurations': configurations,
            'predictions': predictions,
            'best_config': best_config,
            'best_pred': best_pred,
            'Target': data['desire'],
            'filename': data['filename'],
            'erase': erase_cols,
            'pred_all': pred_all
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
'''---------------------------------------------Local 재탐색----------------------------------------------'''
@app.route('/api/rerun_prediction', methods=['POST'])
def rerun_prediction():
    """
    이미 존재하는 results 폴더(=outputs/폴더명)와 
    새로운 starting_points 등을 받아서 다시 parameter_prediction 로직을 수행하는 API
    """
    try:
        data = request.get_json()
        if 'save_name' not in data:
            return jsonify({'status': 'error', 'message': 'save_name이 누락되었습니다.'}), 400

        save_name = data['save_name']
        outputs_dir = 'outputs'
        subfolder_path = os.path.join(outputs_dir, save_name)

        # 기존 폴더 존재 여부 확인
        if not os.path.exists(subfolder_path):
            return jsonify({
                'status': 'error', 
                'message': f'해당 폴더({save_name})가 존재하지 않습니다.'
            }), 400

        # 기존 input.json 불러오기
        input_json_path = os.path.join(subfolder_path, f"{save_name}_input.json")
        if not os.path.exists(input_json_path):
            return jsonify({
                'status': 'error', 
                'message': f'기존 input.json 파일이 {save_name} 폴더 안에 없습니다.'
            }), 400

        with open(input_json_path, 'r', encoding='utf-8') as f:
            old_input_data = json.load(f)

        # old_input_data에 들어있는 값과 새로 들어온 data를 합친다.

        # 우선순위: 새 data 값 > old_input_data 값
        # (기본적으로 old_input_data를 복사한 뒤, 새 data에 있는 키는 덮어쓴다)
        combined_data = old_input_data.copy()
        
        for key, val in data.items():
            # 일부 key만 업데이트, 혹은 모든 key 업데이트 등 원하는 로직에 따라
            combined_data[key] = val
        
        # 예: "starting_points"를 새로 받았으면 덮어씌우기
        #     만약 "desire", "strategy" 등도 바꾸고 싶다면 동일한 방법으로 진행
        # combined_data['starting_points'] = data.get('starting_points', old_input_data.get('starting_points', {}))
        # (4-2) starting_points 업데이트: 요청에 'starting_points'가 있으면 덮어씌움
        if 'starting_points' in data:
            combined_data['starting_points'] = data['starting_points']
        # combined_data['desire'] = float(data.get('desire', old_input_data.get('desire', 0.0)))
        # ...

        # 이제 combined_data를 기반으로 다시 모델 로딩, parameter_prediction 실행
        # (아래 로직은 /api/submit_prediction 에 있는 것을 최대한 재활용)

        # 1) filename, metadata, CSV 로드
        filename = combined_data['filename']
        data_file_path = os.path.join('uploads', filename)
        if not os.path.exists(data_file_path):
            return jsonify({'status': 'error', 'message': f'Data file not found: {filename}'}), 400

        df = pd.read_csv(data_file_path).drop_duplicates()
        
        # 2) 메타데이터 로드
        metadata_path = os.path.join(METADATA_FOLDER, f"{filename}_metadata.json")
        if not os.path.exists(metadata_path):
            return jsonify({
                'status': 'error', 
                'message': f'Metadata file not found for: {filename}'
            }), 400

        with open(metadata_path, 'r', encoding='utf-8') as f:
            stored_metadata = json.load(f)
        units = [item['unit'] for item in stored_metadata]
        lower_bound = [item['min'] for item in stored_metadata]
        upper_bound = [item['max'] for item in stored_metadata]
        round_values = [item['round'] for item in stored_metadata]
        data_type = [item['data_type'] for item in stored_metadata]
        # (원하는 대로 min_boundary, max_boundary, unit 등 로드)
        # 5-4) erase_cols 처리 (변동없는 열 제거)
        erase_cols = []
        erase_indices = []
        feature_cols = [col for col in df.columns if col != 'Target']
        for idx, col in enumerate(feature_cols):
            if df[col].min() == df[col].max():
                erase_cols.append(col)
                erase_indices.append(idx)
        if erase_cols:
            df.drop(columns=erase_cols, inplace=True)
        
        # 5-5) 메타데이터 필터링
        filtered_metadata = [
            item for item in stored_metadata 
            if item['column'] not in erase_cols
        ]
        
        # 5-6) 필터링 후 배열들 재생성
        #      submit_prediction 코드 참조
        def filter_by_indices(original_list, indices_to_remove):
            return [val for idx, val in enumerate(original_list) if idx not in indices_to_remove]

        # 새 units, lower_bound, upper_bound, round_values, data_type, starting_points
        units = []
        max_boundary = []
        min_boundary = []
        decimal_place = []
        data_type_list = []
        
        for item in filtered_metadata:
            units.append(item["unit"])
            max_boundary.append(item["max"])
            min_boundary.append(item["min"])
            decimal_place.append(item["round"])
            data_type_list.append(item["data_type"])

        # dtype 매핑
        data_type_mapping = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool
        }
        data_type_list = [data_type_mapping[dt] for dt in data_type_list]
        # (5-7) starting_point 처리
        #      combined_data['starting_points']는 dict로 가정
        # sp_full = list(combined_data['starting_points'].values())
        # sp_filtered = filter_by_indices(sp_full, erase_indices)
        # 5-8) 모델 로드
        model_names = combined_data['models']  # 기존 input.json 안 models
        models_list = []
        MODEL_FOLDER = 'models'
        # 3) 모델 로딩 로직
        for model_name in model_names:
            model_dir = os.path.join(MODEL_FOLDER, model_name)
            model_metadata_path = os.path.join(model_dir, f"{model_name}.json")
            if not os.path.exists(model_metadata_path):
                logging.warning(f"No metadata found for model: {model_name}")
                continue

            with open(model_metadata_path, 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)

            framework = model_metadata.get('framework')
            model_selected = model_metadata.get('model_selected')
            hyperparams = model_metadata.get('parameters', {})
            input_size = model_metadata.get('input_size', None)

            if framework == 'sklearn':
                pkl_path = os.path.join(model_dir, f"{model_name}.pkl")
                if not os.path.isfile(pkl_path):
                    continue
                model = joblib.load(pkl_path)
                models_list.append(model)

            elif framework == 'pytorch':
                pt_path = os.path.join(model_dir, f"{model_name}.pt")
                model = create_model(model_selected, input_size=input_size, hyperparams=hyperparams)
                if os.path.isfile(pt_path):
                    state_dict = torch.load(pt_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                models_list.append(model)

            else:
                logging.warning(f"Unsupported framework: {framework}")
                continue

        # 4) parameter_prediction 실행에 필요한 인자 셋팅
        #    (원하는 로직에 맞게)
        
        starting_points = combined_data['starting_points']  # 예: dict 형태
        desired = float(combined_data['desire'])
        mode = combined_data['option']  # 'local' or 'global'
        modeling = combined_data['modeling_type'].lower()
        strategy = combined_data['strategy'].lower()
        tolerance = combined_data.get('tolerance', None)
        beam_width = combined_data.get('beam_width', None)
        num_candidates = combined_data.get('num_candidates', None)
        escape = combined_data.get('escape', True)
        top_k = combined_data.get('top_k', 2)
        index = combined_data.get('index', 0)
        up = combined_data.get('up', True)
        alternative = combined_data.get('alternative', 'keep_move')
        # 5-10) df.max()/min() 이용해서 boundary 업데이트 (submit_prediction과 동일)
        upper_bound = df.max().tolist()
        lower_bound = df.min().tolist()
        
        # starting_points가 dictionary일 경우, 모델에 들어갈 때 list나 numpy 변환 등이 필요할 수 있음
        # 예를 들어:
        # sp_list = [float(v) for v in starting_points.values()]  # 순서주의

        # 실제 parameter_prediction 호출
        # (아래는 /api/submit_prediction 예시를 그대로 차용)

        # 5-11) parameter_prediction 호출
        
        configurations, predictions, best_config, best_pred, pred_all = parameter_prediction(
            data=df,
            models=models_list,
            desired=desired,
            starting_point=starting_points, 
            mode=mode,
            modeling=modeling,
            strategy=strategy,
            tolerance=tolerance, 
            beam_width=beam_width,
            num_candidates=num_candidates,
            escape=escape, 
            top_k=top_k,
            index=index,
            up=up,
            alternative=alternative,
            unit=units,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            data_type=data_type_list,
            decimal_place=decimal_place
        )

        # (6) 결과 저장
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        combined_data['timestamp'] = timestamp

        # output_data 생성
        configurations = convert_to_python_types(configurations)
        predictions = convert_to_python_types(predictions)
        best_config = convert_to_python_types(best_config)
        best_pred = convert_to_python_types(best_pred)
        pred_all = convert_to_python_types(pred_all)

        output_data = {
            'mode': mode,
            'timestamp': timestamp,
            'configurations': configurations,
            'predictions': predictions,
            'best_config': best_config,
            'best_pred': best_pred,
            'Target': desired,
            'filename': filename,
            'erase': erase_cols,
            'pred_all': pred_all
        }
        print(best_config)
        # (6-1) 새로운 input.json, output.json 저장
        new_input_path = os.path.join(subfolder_path, f"{save_name}_input.json")
        with open(new_input_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)

        new_output_path = os.path.join(subfolder_path, f"{save_name}_output.json")
        with open(new_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        return jsonify({'status': 'success', 'data': output_data}), 200

    except Exception as e:
        logging.error(f"An error occurred in rerun_prediction: {str(e)}")
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
        input_file_path = os.path.join(folder_path, f"{folder_name}_input.json")  # <-- (1) input JSON 경로도 준비
        output_file_path = os.path.join(folder_path, f"{folder_name}_output.json")
        
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                    # (2) hyperparameter용 데이터를 저장할 변수
                    hyperparams_data = {}

                    # (3) input.json 파일이 있으면 로드
                    if os.path.exists(input_file_path):
                        try:
                            with open(input_file_path, 'r', encoding='utf-8') as fin:
                                hyperparams_data = json.load(fin)
                        except Exception as e:
                            print(f"Error reading {input_file_path}: {e}")
                            hyperparams_data = {}
                    results.append({
                        'mode': output_data.get('mode'),
                        'timestamp': output_data.get('timestamp'),
                        'target': output_data.get('Target'),
                        'folder_name': str(folder_name),
                        'predictions' : output_data.get('predictions'),
                        'configurations': output_data.get('configurations'),
                        'best_config': output_data.get('best_config'),
                        'best_pred': output_data.get('best_pred'),
                        'hyperparameter': hyperparams_data,
                        'erase': output_data.get('erase'),
                        'pred_all':output_data.get('pred_all')

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


# 사용자 제공했던 EarlyStopping 클래스 그대로 사용
class EarlyStopping:
    def __init__(self, patience=10, delta=0.1):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = float('inf')

    def __call__(self, train_loss):
        score = -train_loss
        if self.best_score is None:
            self.best_score = score
            self.train_loss_min = train_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.train_loss_min = train_loss
def load_constraints_from_metadata(csv_filename):
    """
    csv_filename: 예) "mydata.csv" (확장자 포함)
                  -> 메타데이터 파일 이름은 "mydata.csv_metadata.json" 이라 가정

    메타데이터 JSON 예시:
    [
      {
        "column": "att1",
        "unit": 5,
        "min": 0,
        "max": 100
      },
      {
        "column": "att2",
        "unit": 10,
        "min": -5,
        "max": 50
      }
    ]
    """

    # 메타데이터 파일 경로
    metadata_path = os.path.join(METADATA_FOLDER, f"{csv_filename}_metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    # constraints를 담을 딕셔너리
    constraints = {}

    # 예) round_digits를 어떻게 설정할지 미리 정하거나, 메타데이터에 추가 필드가 있다면 거기서 읽어올 수도 있음
    # 여기서는 "float형이면 round_digits=2, int형이면 round_digits=0" 같은 식으로 단순 예시
    # 혹은 column 이름에 따라 결정할 수도 있음
    def infer_type_and_digits(column_name, unit_val):
        # 예시: unit값이 정수이면 int, 아니면 float로 가정
        if float(unit_val).is_integer():
            return int, 0  # (타입, 반올림 자릿수=0)
        else:
            return float, 2  # (타입, 반올림 자릿수=2)

    for item in metadata_list:
        col_name = item['column']      # 예: "att1"
        col_unit = float(item['unit']) # 메타데이터에 저장된 unit
        col_min  = float(item['min'])
        col_max  = float(item['max'])

        # 타입과 반올림 자릿수를 임의 규칙(혹은 추가 메타데이터)에 따라 결정
        inferred_type, round_digits = infer_type_and_digits(col_name, col_unit)

        # 이제 [단위값, min, max, 타입, 반올림자릿수] 형태로 constraints 구성
        constraints[col_name] = [col_unit, col_min, col_max, inferred_type, round_digits]

    return constraints




def rae(y_true, y_pred):
        numerator = np.sum(np.abs(y_pred - y_true))
        denominator = np.sum(np.abs(y_true - np.mean(y_true)))
        return numerator / denominator if denominator != 0 else np.nan


def convert_to_python_types(obj):
    """입력된 obj(리스트, 스칼라 등) 내 모든 원소를 파이썬 기본 자료형으로 변환"""
    if isinstance(obj, torch.Tensor):
        # 텐서를 파이썬 list로
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        # 넘파이 배열 -> list
        return obj.tolist()
    elif isinstance(obj, list):
        # 리스트라면, 내부 요소를 재귀적으로 변환
        return [convert_to_python_types(o) for o in obj]
    elif isinstance(obj, (float, int, str)):
        # 기본형이면 그대로 리턴
        return obj
    else:
        # 혹시 모르는 케이스 (ex: np.float32 등)은 float()로 캐스팅
        try:
            return float(obj)
        except:
            # 그래도 안 되면 그냥 문자열 처리
            return str(obj)
        
if __name__ == '__main__':
    app.run(debug=True)