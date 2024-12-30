from flask import Flask, request, jsonify, send_from_directory, render_template, make_response
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os, datetime
import random
import math
import pandas as pd
import mimetypes
import pickle
import json
from model import create_model
# from data_preprocessing import MinMaxScaling
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

# CORS(app)
# 특정 Origin 허용
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173","http://localhost:5173/", "http://127.0.0.1:5173/"]}})

# 업로드 및 모델 저장 디렉토리 설정
mimetypes.init()
mimetypes.add_type('application/javascript', '.js', strict=True)

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
    input_size = data.get('input_size', None)
    csv_filename = data.get('csv_filename')
    hyperparams = data.get('hyperparameters', {})
    target_column = "Target"
    
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

    # 모델 생성
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

    # 폴더 생성
    
    model_path = os.path.join(save_dir, f"{model_name}.pt" if isinstance(model, torch.nn.Module) else f"{model_name}.pkl")
    metadata_path = os.path.join(save_dir, f"{model_name}.json")
    creation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 데이터 전처리
    df = df.fillna(0)
    # 여기서 X, y를 정의
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 사이킷런 MinMaxScaler 사용
    from sklearn.preprocessing import MinMaxScaler

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # X와 y를 numpy 형태로 변환하여 fit_transform
    X = X.to_numpy()  # DataFrame -> Numpy array
    y = y.to_numpy().reshape(-1, 1)  # y를 2D array로
    print(X.shape, y.shape)
    # 스케일링 적용
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).flatten()

    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=val_ratio, random_state=42)

    # feature, target을 np.array 또는 torch.tensor 형태로 변환
    # Sklearn 모델: np.array 그대로 사용
    # Pytorch 모델: tensor로 바꾸는 건 _train_nn 내부에서 할 수도 있음
    X_train_np = X_train
    y_train_np = y_train.reshape(-1,1)
    X_val_np = X_val
    y_val_np = y_val.reshape(-1,1)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    # 모델 학습
    try:
        #print("모델 학습 시작...")
        if isinstance(model, torch.nn.Module):
            training_losses = _train_nn(model, X_train_np, y_train_np)
        else:
            model.fit(X_train_np, y_train_np)  # 여기서 에러 발생 가능성 확인
        #print("모델 학습 완료.")
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        return jsonify({"status": "error", "message": f"모델 학습 중 오류 발생: {str(e)}"}), 500
    


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
            loaded_model = create_model(...)  # 모델 구조 생성
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


    def rae(y_true, y_pred):
        numerator = np.sum(np.abs(y_pred - y_true))
        denominator = np.sum(np.abs(y_true - np.mean(y_true)))
        return numerator / denominator if denominator != 0 else np.nan

    train_mse = mean_squared_error(y_train_np, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_rae = rae(y_train_np, train_predictions)

    if val_predictions is not None:
        val_mse = mean_squared_error(y_val_np, val_predictions)
        val_rmse = np.sqrt(val_mse)
        val_rae = rae(y_val_np, val_predictions)
    else:
        val_mse = val_rmse = val_rae = None

    # 비정규화 (y 값만 역변환)
    y_train_inv = scaler_y.inverse_transform(y_train_np)
    y_val_inv = scaler_y.inverse_transform(y_val_np) if y_val_np is not None else None
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
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"메타데이터 저장 중 오류 발생: {str(e)}"}), 500

    # 결과 반환
    result = {
        'status': 'success',
        'message': f'모델 {model_name} 생성 및 학습 완료',
        'model_name': model_name,
        'csv_filename': csv_filename,
        'creation_time': creation_time,
        'hyperparameters': hyperparams,
        'metrics': model_info['metrics']
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
        # model_list = process_models(data['models'])
        
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
        
        # ============================
        # 여기서부터 모델 생성/파라미터 로드 로직
        # ============================
        MODEL_FOLDER = 'models'
        models_list = []

        # for model_name in data['models']:
        #     model_dir = os.path.join(MODEL_FOLDER, model_name)
        #     metadata_path = os.path.join(model_dir, f"{model_name}.json")
        #     model_path = os.path.join(model_dir, f"{model_name}.pkl")
            
        #     if not os.path.exists(metadata_path):
        #         logging.error(f"No metadata found for model: {model_name}")
        #         continue

        #     try:
        #         # 메타데이터 로드
        #         with open(metadata_path, 'r', encoding='utf-8') as f:
        #             metadata = json.load(f)

        #         framework = metadata.get('framework')
        #         model_selected = metadata.get('model_selected')
        #         hyperparams = metadata.get('parameters', {})
        #         input_size = metadata.get('input_size', None)
        #         model_path = ""
        #         print(model_selected)
        #         # framework에 따라 model_path 설정
        #         if framework == 'sklearn':
        #             model_path = os.path.join(model_dir, f"{model_name}.pkl")
        #         elif framework == 'pytorch':
        #             model_path = os.path.join(model_dir, f"{model_name}.pt")
        #         else:
        #             logging.error(f"Unsupported framework '{framework}' for model: {model_name}")
        #             continue
        #         if framework == 'sklearn':
        #             # (1) pkl 파일이 존재하면 바로 불러오기
        #             try:
        #                 with open(model_path, 'rb') as pf:
        #                     model = joblib.load(pf)
        #                 print(f"모델 로드 성공 (pkl): {model}")
        #             except Exception as e:
        #                 logging.error(f"Error while loading model '{model_name}' from pkl: {e}")
        #                 continue

        #         elif framework == 'pytorch':
        #             # PyTorch 모델 로드 로직 (기존 그대로 유지)
        #             model = create_model(model_selected, input_size=input_size, hyperparams=hyperparams)
        #             if not model:
        #                 raise ValueError(f"PyTorch 모델 생성 실패: {model_name}")
        #             print(f"PyTorch 모델 생성 성공: {model}")

        #             if os.path.isfile(model_path):
        #                 state_dict = torch.load(model_path, map_location='cpu')
        #                 model.load_state_dict(state_dict)
        #                 print(f"PyTorch 모델 파라미터 로드 성공: {model}")

        #         else:
        #             raise ValueError(f"지원되지 않는 프레임워크입니다: {framework}")

        #         # 완성된 모델 리스트에 추가
        #         models_list.append(model)

        #     except Exception as e:
        #         logging.error(f"Error while creating/loading model '{model_name}': {e}")
        #         continue

        # print("모델 생성/로딩 완료:")
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

        # 파일 경로 설정 (폴더명 기반 파일 이름 생성)
        input_file_path = os.path.join(subfolder_path, f'{save_name}_input.json')
        output_file_path = os.path.join(subfolder_path, f'{save_name}_output.json')

        # models, training_losses, configurations, predictions, best_config, best_pred = run(
        #     data=df,  # DataFrame
        #     models=models,
        #     model_list=model_list,
        #     desired=float(desired),  # 실수형으로 변환
        #     starting_point=[float(value) for value in data['starting_points'].values()],  # 숫자 리스트로 변환
        #     mode=mode,
        #     modeling=modeling.lower(),
        #     strategy=strategy.lower(),
        #     tolerance=float(tolerance) if tolerance is not None else None,  # 실수형 변환
        #     beam_width=int(beam_width) if beam_width is not None else None,
        #     num_cadidates=int(num_candidates),
        #     escape=bool(escape),
        #     top_k=int(top_k),
        #     index=int(index),
        #     up=bool(up),
        #     alternative=alternative
        # )
        

        # 현재 시각(날짜) 정보
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # 입력 데이터에도 날짜정보 추가
        data['timestamp'] = timestamp

        # option에 따라 50개(global) 혹은 10개(local) 생성
        n_points = 50 if data['option'] == 'global' else 10

        # 입력값들(변수들) 키 목록 (딕셔너리이므로 순서를 맞추기 위해 sorted 사용 가능)
        var_keys = sorted(data['starting_points'].keys())

        # configurations & predictions 초기화
        configurations = []
        predictions = []

        # 정규분포 생성을 위한 임시 표준편차(예시): 입력값 생성용 stdev=1.0, 예측값 생성용 stdev=10.0
        stdev_for_config = 1.0
        stdev_for_pred = 10.0

        target_value = float(data['desire'])

        # (1) configurations 생성
        for _ in range(n_points):
            row = []
            for var in var_keys:
                start_val = float(data['starting_points'][var])
                unit_val = float(data['units'][var])
                # 정규분포에서 샘플
                sampled = random.gauss(start_val, stdev_for_config)
                # unit단위 반올림
                if unit_val != 0:
                    sampled = round(sampled / unit_val) * unit_val
                row.append(sampled)
            configurations.append(row)

        # (2) predictions 생성
        #    Target값을 중심으로 정규분포 샘플링해서 n_points 개 생성
        for _ in range(n_points):
            pred = random.gauss(target_value, stdev_for_pred)
            predictions.append(pred)

        # (3) best_config / best_pred 찾기
        #    각 predictions에 대해 Target과의 차(diffs) 계산
        diffs = [abs(p - target_value) for p in predictions]

        if data['option'] == 'local':
            # local인 경우: (config, pred, diff)를 하나의 튜플로 묶어서 오름차순 정렬 후 반환
            combined = list(zip(configurations, predictions, diffs))
            combined.sort(key=lambda x: x[2])  # diff 기준 정렬

            # 정렬된 결과를 다시 분리
            configurations = [c[0] for c in combined]
            predictions = [c[1] for c in combined]
            diffs_sorted = [c[2] for c in combined]

            # 가장 앞(오차가 작은) 것이 best
            best_idx = 0
            best_config = configurations[best_idx]
            best_pred = predictions[best_idx]

        else:
            # global인 경우: 정렬 없이, 오차가 가장 적은 인덱스를 찾는다
            best_idx = min(range(n_points), key=lambda i: diffs[i])
            best_config = configurations[best_idx]
            best_pred = predictions[best_idx]
        output_data = {
            'mode': data['option'],
            'timestamp': timestamp,  # output에도 동일 날짜 정보
            'configurations': configurations,
            'predictions': predictions,
            'best_config': best_config,
            'best_pred': best_pred,
            'Target': data['desire'],
            'filename': data['filename'],
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
                        'mode': output_data.get('mode'),
                        'timestamp': output_data.get('timestamp'),
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



def _train_nn(model, X_train, y_train, epochs=100, lr=0.001, batch_size=32):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=10)

    # NumPy 배열을 PyTorch 텐서로 변환
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    training_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # NaN 또는 Inf 체크
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"손실 값에 문제가 발생했습니다. Loss: {loss.item()}")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        training_losses.append(avg_loss)
        print(f"[DEBUG] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Early stopping 체크
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f"[DEBUG] Early stopping triggered at epoch {epoch+1}")
            break

    return training_losses
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

if __name__ == '__main__':
    app.run(debug=True)