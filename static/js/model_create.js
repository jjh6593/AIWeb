document.addEventListener('DOMContentLoaded', function () {
    // 모델 생성 및 업로드 페이지 기능
    const modelForm = document.getElementById('modelForm');
    const uploadModelForm = document.getElementById('uploadModelForm');

    const modelOptions = {
        pytorch: ['MLP_1', 'MLP_2', 'MLP_3'],
        scikit: [
            'LinearRegressor', 'Ridge', 'Lasso', 'ElasticNet',
            'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
            'SVR', 'KNeighborsRegressor', 'HuberRegressor', 'GaussianProcessRegressor', 'XGBoost'
        ]
    };

    const modelTypeSelect = document.getElementById('modelType');
    const modelListDropdown = document.getElementById('modelListDropdown');
    const inputSizeField = document.getElementById('inputSizeField');
    const csvSelect = document.getElementById('csvFileSelect1'); // CSV 파일 선택 드롭다운
    const trainButton = document.createElement('button'); // 학습 버튼
    trainButton.textContent = '학습 시작';
    trainButton.classList.add('btn', 'btn-primary', 'mt-3');
    // 검증 데이터 비율 설정 라벨 추가
    const valRatioLabel = document.createElement('label');
    valRatioLabel.classList.add('form-label', 'mt-3');
    valRatioLabel.setAttribute('for', 'valRatio');
    valRatioLabel.textContent = '검증 데이터 비율 (0~1)';
    
    const valRatioInput = document.createElement('input'); // 검증 데이터 비율 입력 필드
    valRatioInput.type = 'number';
    valRatioInput.classList.add('form-control', 'mt-3');
    valRatioInput.id = 'valRatio';
    valRatioInput.name = 'val_ratio';
    valRatioInput.min = 0;
    valRatioInput.max = 1;
    valRatioInput.step = 0.1;
    valRatioInput.value = 0.2;


    // 학습 버튼과 입력 필드를 추가할 컨테이너
    const container = csvSelect.parentElement;
    container.appendChild(valRatioLabel);
    container.appendChild(valRatioInput);
    container.appendChild(trainButton);
    
    // CSV 파일 목록 불러오기
    fetch('/api/get_csv_files')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                csvSelect.innerHTML = ''; // 드롭다운 초기화
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.textContent = file;
                    csvSelect.appendChild(option);
                });
            } else {
                alert('CSV 파일 목록을 불러오는 데 실패했습니다.');
            }
        });

    // 학습 버튼 클릭 이벤트 처리
    trainButton.addEventListener('click', function () {
        const csvFilename = csvSelect.value;
        const valRatio = parseFloat(valRatioInput.value);

        if (!csvFilename || isNaN(valRatio)) {
            alert('CSV 파일을 선택하고 검증 데이터 비율을 입력하세요.');
            return;
        }
        const modelType = modelTypeSelect.value;
        const modelSelected = modelListDropdown.value;
        // RandomForestRegressor와 GradientBoostingRegressor는 아직 지원되지 않는 기능
        if (modelSelected === 'RandomForestRegressor' || modelSelected === 'GradientBoostingRegressor') {
            alert('RandomForest와 GradientBossting은 아직 지원이 되지 않는 기능입니다.');
            
            modelListDropdown.value = ''; // 선택 해제
            return; // 이후 기능 실행하지 않음
        }
        const modelName = document.getElementById('modelName').value;
        const inputSize = document.getElementById('inputSize').value;

        if (!modelType || !modelSelected || !modelName) {
            alert('모델 타입, 모델 선택, 그리고 모델 이름을 입력하세요.');
            return;
        }
        const modelInfo = {
            model_type: modelType,
            model_selected: modelSelected,
            model_name: modelName,
            input_size: modelType === 'pytorch' ? parseInt(inputSize) : null
        };
        
            // 1. 모델 생성 요청
            fetch('/api/save_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(modelInfo)
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(`모델 생성 성공: ${data.message}`);
                        // loadModelList();

                        // 2. 모델 학습 요청
                        const payload = {
                            csv_filename: csvFilename,
                            model_name: modelName,
                            target_column : 'Target',
                            val_ratio: valRatio
                        };

                        fetch('/api/train_model', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        })
                            .then(response => response.json())
                            .then(data => {
                                if (data.status === 'success') {
                                    alert('모델 학습이 성공적으로 완료되었습니다!');
                                    console.log('Training Result:', data.result); // 학습 결과 로그 출력
                                } else {
                                    alert(`모델 학습 중 오류 발생: ${data.message}`);
                                }
                            })
                            .catch(error => {
                                console.error('Error during training:', error);
                                alert('모델 학습 중 오류가 발생했습니다.');
                            });
                    } else {
                        alert(`모델 생성 중 오류 발생: ${data.message}`);
                    }
                })
                .catch(error => {
                    console.error('Error during model creation:', error);
                    alert('모델 생성 중 오류가 발생했습니다.');
                });
            });
    if (modelTypeSelect) {
        modelTypeSelect.addEventListener('change', function () {
            const selectedType = modelTypeSelect.value;

            if (selectedType === 'pytorch') {
                inputSizeField.style.display = 'block';
            } else {
                inputSizeField.style.display = 'none';
            }

            modelListDropdown.innerHTML = '';

            if (selectedType && modelOptions[selectedType]) {
                modelOptions[selectedType].forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelListDropdown.appendChild(option);
                });

                modelListDropdown.disabled = false;
            } else {
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = '모델 타입을 먼저 선택하세요';
                modelListDropdown.appendChild(defaultOption);
                modelListDropdown.disabled = true;
            }
        });
    }

    if (modelForm) {
        modelForm.addEventListener('submit', function (event) {
            event.preventDefault();
            const modelType = modelTypeSelect.value;
            const modelSelected = modelListDropdown.value;
            const modelName = document.getElementById('modelName').value;
            const inputSize = document.getElementById('inputSize').value;

            if (!modelType || !modelSelected || !modelName) {
                alert('모델 타입, 모델 선택, 그리고 모델 이름을 입력하세요.');
                return;
            }

            const modelInfo = {
                model_type: modelType,
                model_selected: modelSelected,
                model_name: modelName,
                input_size: modelType === 'pytorch' ? parseInt(inputSize) : null
            };

            fetchJSON('/api/save_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(modelInfo)
            }, data => {
                alert(data.message);
                loadModelList();
            });
        });
    }

    if (uploadModelForm) {
        uploadModelForm.addEventListener('submit', function (event) {
            event.preventDefault();

            const modelFileInput = document.getElementById('uploadedModel');
            const modelNameInput = document.getElementById('uploadedModelName');

            const modelFile = modelFileInput.files[0];
            const modelName = modelNameInput.value;

            if (!modelFile || !modelName) {
                alert('모델 파일과 이름을 입력하세요.');
                return;
            }

            const formData = new FormData();
            formData.append('model_file', modelFile);
            formData.append('model_name', modelName);

            fetchJSON('/api/upload_model', { method: 'POST', body: formData }, data => {
                if (data.status === 'success') {
                    alert(data.message);
                    loadModelList();
                } else {
                    alert(data.message);
                }
            });
        });
    }
    
    function loadModelList() {
        const modelList = document.getElementById('modelList');
        if (modelList) {
            fetchModels(models => {
                modelList.innerHTML = '';
                models.forEach(model => {
                    const li = document.createElement('li');
                    li.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
                    li.textContent = `${model.model_name} (${model.model_selected})`;

                    const deleteButton = document.createElement('button');
                    deleteButton.classList.add('btn', 'btn-danger', 'btn-sm');
                    deleteButton.textContent = '삭제';
                    deleteButton.addEventListener('click', function () {
                        if (confirm(`모델 ${model.model_name}을(를) 삭제하시겠습니까?`)) {
                            fetchJSON('/api/delete_model', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ model_name: model.model_name })
                            }, data => {
                                alert(data.message);
                                loadModelList();
                            });
                        }
                    });

                    li.appendChild(deleteButton);
                    modelList.appendChild(li);
                });
            });
        }
    }

    loadModelList();
});
