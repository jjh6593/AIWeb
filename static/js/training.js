document.addEventListener('DOMContentLoaded', function() {
    const trainingForm = document.getElementById('trainingForm');
    const csvSelect = document.getElementById('csvFileSelect');
    const modelSelect = document.getElementById('modelSelect');
    const addModelButton = document.getElementById('addModelButton');
    const modelList = document.getElementById('modelList');
    const trainingProgress = document.getElementById('trainingProgress');
    let selectedModels = [];

    // CSV 파일 및 모델 목록 로드
    function loadOptions() {
        fetch('/api/get_csv_files')
            .then(response => response.json())
            .then(data => {
                csvSelect.innerHTML = '';
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.textContent = file;
                    csvSelect.appendChild(option);
                });
            });

        fetch('/api/get_models')
            .then(response => response.json())
            .then(data => {
                modelSelect.innerHTML = '';
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.model_name;
                    option.textContent = model.model_name;
                    modelSelect.appendChild(option);
                });
            });
    }

    // 모델 추가 버튼 클릭 이벤트
    addModelButton.addEventListener('click', function() {
        const modelName = modelSelect.value;
        if (!modelName) {
            alert('모델을 선택하세요.');
            return;
        }
        if (selectedModels.includes(modelName)) {
            alert('이미 선택된 모델입니다.');
            return;
        }
        selectedModels.push(modelName);
        updateModelList();
    });

    // 선택된 모델 리스트 업데이트
    function updateModelList() {
        modelList.innerHTML = '';
        selectedModels.forEach(modelName => {
            const li = document.createElement('li');
            li.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');

            const modelInfo = document.createElement('div');
            modelInfo.innerHTML = `<strong>${modelName}</strong><br><small>Training Loss: N/A | Validation Loss: N/A</small>`;
            modelInfo.id = `model-info-${modelName}`;

            const removeButton = document.createElement('button');
            removeButton.classList.add('btn', 'btn-danger', 'btn-sm');
            removeButton.textContent = '제거';
            removeButton.addEventListener('click', function() {
                selectedModels = selectedModels.filter(name => name !== modelName);
                updateModelList();
            });

            li.appendChild(modelInfo);
            li.appendChild(removeButton);
            modelList.appendChild(li);
        });
    }

    // 모델 학습 폼 제출 이벤트
    trainingForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const csvFilename = csvSelect.value;
        const targetColumn = document.getElementById('targetColumn').value;
        const valRatio = parseFloat(document.getElementById('valRatio').value);

        if (!csvFilename || !targetColumn || selectedModels.length === 0) {
            alert('모든 필드를 입력하고 모델을 추가하세요.');
            return;
        }

        // 각 모델에 대해 학습 진행
        (async function trainModels() {
            for (const modelName of selectedModels) {
                await trainModel(csvFilename, modelName, targetColumn, valRatio);
            }
            alert('모든 모델의 학습이 완료되었습니다.');
        })();
    });

    // 모델 학습 함수
    async function trainModel(csvFilename, modelName, targetColumn, valRatio) {
        return new Promise((resolve, reject) => {
            const trainingData = {
                csv_filename: csvFilename,
                model_name: modelName,
                target_column: targetColumn,
                val_ratio: valRatio
            };

            fetch('/api/train_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trainingData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('네트워크 응답에 문제가 있습니다.');
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    updateModelInfo(modelName, data.result);
                    resolve();
                } else {
                    alert(`모델 ${modelName} 학습 중 오류 발생: ${data.message}`);
                    resolve();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                resolve();
            });
        });
    }

    // 모델 정보 업데이트
    function updateModelInfo(modelName, result) {
        const modelInfo = document.getElementById(`model-info-${modelName}`);
        if (modelInfo) {
            modelInfo.innerHTML = `<strong>${modelName}</strong><br><small>Training Loss: ${result.train_loss.toFixed(4)} | Validation Loss: ${result.val_loss ? result.val_loss.toFixed(4) : 'N/A'}</small>`;
        }
    }

    // 초기 옵션 로드
    loadOptions();
});

// static/js/training.js
document.addEventListener('DOMContentLoaded', function() {
    const modelSelect = document.getElementById('modelSelect');

    // 모델 옵션 불러오기 함수
    function loadModelOptions() {
        fetch('/api/get_models')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.models.length > 0) {
                    modelSelect.innerHTML = ''; // 기존 옵션 제거
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.model_name;
                        option.textContent = `${model.model_name} (${model.framework})`;
                        modelSelect.appendChild(option);
                    });
                } else {
                    alert('저장된 모델이 없습니다.');
                }
            })
            .catch(error => console.error('모델 목록 로드 중 오류:', error));
    }

    // 초기화 시 모델 옵션 로드
    loadModelOptions();
});
