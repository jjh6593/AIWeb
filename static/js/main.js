// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    // 공통으로 사용되는 변수들
    let uploadedFilename = '';

    // 요소 선택
    const uploadForm = document.getElementById('uploadForm');
    const loadFileButton = document.getElementById('loadFileButton');
    const dataSettingsForm = document.getElementById('dataSettingsForm');
    const dataPreview = document.getElementById('dataPreview');
    const excludeColumnsForm = document.getElementById('excludeColumnsForm');
    const saveFilteredFileButton = document.getElementById('saveFilteredFile');
    const newFilenameInput = document.getElementById('newFilename');
    const trainingForm = document.getElementById('trainingForm');
    const modelForm = document.getElementById('modelForm');
    const uploadModelForm = document.getElementById('uploadModelForm');
    const loadFileModalElemet = document.getElementById('loadFileModal');
    const loadFileModal = new bootstrap.Modal(loadFileModalElemet);
    
    // 공통 함수 정의
    function fetchJSON(url, options = {}, callback) {
        fetch(url, options)
            .then(response => response.json())
            .then(data => callback(data))
            .catch(error => console.error('Error:', error));
    }

    function fetchCSVFiles(callback) {
        fetchJSON('/api/get_csv_files', {}, data => {
            if (data.status === 'success') {
                callback(data.files);
            } else {
                alert(data.message);
            }
        });
    }

    function fetchModels(callback) {
        fetchJSON('/api/get_models', {}, data => {
            if (data.models) {
                callback(data.models);
            } else {
                alert('모델 목록을 불러오지 못했습니다.');
            }
        });
    }

    // CSV 미리보기 및 컬럼 제외 기능
    function loadCSVPreview(filename) {
        fetchJSON(`/api/get_csv_data?filename=${filename}`, {}, data => {
            if (data.status === 'success') {
                renderDataPreview(data);
                renderExcludeColumns(data);
                renderTargetColumnOptions(data);
            } else {
                alert(data.message);
            }
        });
    }

    function renderDataPreview(data) {
        if (dataPreview) {
            const table = document.createElement('table');
            table.classList.add('table', 'table-striped');

            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            data.columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            data.data_preview.forEach(row => {
                const tr = document.createElement('tr');
                data.columns.forEach(col => {
                    const td = document.createElement('td');
                    td.textContent = row[col];
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);

            dataPreview.innerHTML = '';
            dataPreview.appendChild(table);
        }
    }

    function renderExcludeColumns(data) {
        const excludeColumns = document.getElementById('excludeColumns');
        if (excludeColumns) {
            excludeColumns.innerHTML = '';
            data.columns.forEach(col => {
                const div = document.createElement('div');
                div.classList.add('col-md-3', 'form-check');

                const input = document.createElement('input');
                input.type = 'checkbox';
                input.classList.add('form-check-input');
                input.id = `exclude-${col}`;
                input.value = col;

                const label = document.createElement('label');
                label.classList.add('form-check-label');
                label.setAttribute('for', `exclude-${col}`);
                label.textContent = col;

                div.appendChild(input);
                div.appendChild(label);
                excludeColumns.appendChild(div);
            });
        }
    }

    function renderTargetColumnOptions(data) {
        const targetColumn = document.getElementById('targetColumn');
        if (targetColumn) {
            targetColumn.innerHTML = '';
            data.columns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                targetColumn.appendChild(option);
            });
        }
    }

    // 파일 업로드 및 불러오기 기능
    if (uploadForm || loadFileButton) {
        if (uploadForm) {
            uploadForm.addEventListener('submit', function(event) {
                event.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                fetchJSON('/api/upload_csv', { method: 'POST', body: formData }, data => {
                    if (data.status === 'success') {
                        uploadedFilename = data.filename;
                        alert(data.message);
                        loadCSVPreview(data.filename);
                    } else {
                        alert(data.message);
                    }
                });
            });
        }

        if (loadFileButton) {
            loadFileButton.addEventListener('click', function() {
                fetchCSVFiles(files => {
                    const serverFileList = document.getElementById('serverFileList');
                    if (serverFileList) {
                        serverFileList.innerHTML = '';
                        files.forEach(file => {
                            const listItem = document.createElement('li');
                            listItem.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
                            listItem.textContent = file;

                            const loadButton = document.createElement('button');
                            loadButton.textContent = '불러오기';
                            loadButton.classList.add('btn', 'btn-sm', 'btn-primary');
                            loadButton.addEventListener('click', function() {
                                uploadedFilename = file;
                                loadCSVPreview(file);
                                alert(`${file} 파일을 불러왔습니다.`);
                                loadFileModal.hide();
                            });

                            listItem.appendChild(loadButton);
                            serverFileList.appendChild(listItem);
                        });

                        loadFileModal.show();
                    }
                });
            });
        }

        if (saveFilteredFileButton) {
            saveFilteredFileButton.addEventListener('click', () => {
                const checkedColumns = Array.from(document.querySelectorAll('#excludeColumns input:checked')).map(input => input.value);
                const newFilename = newFilenameInput.value;

                if (!newFilename) {
                    alert('새 파일 이름을 입력하세요.');
                    return;
                }

                if (!uploadedFilename) {
                    alert('업로드된 파일이 없습니다. 파일을 업로드하거나 불러와 주세요.');
                    return;
                }

                const payload = {
                    exclude_columns: checkedColumns,
                    new_filename: newFilename,
                    filename: uploadedFilename
                };

                fetchJSON('/api/save_filtered_csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                }, data => {
                    if (data.status === 'success') {
                        alert(data.message);
                        uploadedFilename = newFilename;
                        loadCSVPreview(newFilename);
                    } else {
                        alert(data.message);
                    }
                });
            });
        }

        if (excludeColumnsForm) {
            excludeColumnsForm.addEventListener('submit', function(event) {
                event.preventDefault();
            });
        }

        if (newFilenameInput) {
            newFilenameInput.addEventListener('keydown', function(event) {
                const inputValue = this.value.trim();
                if (event.key === 'Enter' && inputValue !== '') {
                    event.preventDefault();
                    saveFilteredFileButton.click();
                }
            });
        }
    }

    // 데이터 설정 페이지 기능
    if (dataSettingsForm) {
        document.getElementById('submitSettings').addEventListener('click', function() {
            if (!uploadedFilename) {
                alert('업로드된 파일이 없습니다. 파일을 업로드하거나 불러와 주세요.');
                return;
            }

            const settings = new FormData(dataSettingsForm);
            settings.append('filename', uploadedFilename);
            settings.append('scaler', document.getElementById('scaler').value);
            settings.append('missingHandling', document.getElementById('missingHandling').value);
            settings.append('targetColumn', document.getElementById('targetColumn').value);

            fetchJSON('/api/submit_data_settings', { method: 'POST', body: settings }, data => {
                if (data.status === 'success') {
                    alert(data.message);
                    window.location.href = '/model_creation.html';
                } else {
                    alert(data.message);
                }
            });
        });
    }

    // 모델 생성 및 업로드 페이지 기능
    if (modelForm || uploadModelForm) {
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

        if (modelTypeSelect) {
            modelTypeSelect.addEventListener('change', function() {
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
            modelForm.addEventListener('submit', function(event) {
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
            uploadModelForm.addEventListener('submit', function(event) {
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
                        deleteButton.addEventListener('click', function() {
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
    }

    // 모델 학습 페이지 기능
    if (trainingForm) {
        const csvSelect = document.getElementById('csvFileSelect');
        const modelSelect = document.getElementById('modelSelect');

        function loadOptions() {
            if (csvSelect) {
                fetchCSVFiles(files => {
                    csvSelect.innerHTML = '';
                    files.forEach(file => {
                        const option = document.createElement('option');
                        option.value = file;
                        option.textContent = file;
                        csvSelect.appendChild(option);
                    });
                });
            }

            if (modelSelect) {
                fetchModels(models => {
                    modelSelect.innerHTML = '';
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.model_name;
                        option.textContent = model.model_name;
                        modelSelect.appendChild(option);
                    });
                });
            }
        }

        trainingForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const csvFilename = csvSelect.value;
            const modelName = modelSelect.value;
            const targetColumn = document.getElementById('targetColumn').value;

            if (!csvFilename || !modelName || !targetColumn) {
                alert('모든 필드를 입력하세요.');
                return;
            }

            const trainingData = {
                csv_filename: csvFilename,
                model_name: modelName,
                target_column: targetColumn
            };

            fetchJSON('/api/train_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trainingData)
            }, data => {
                alert(data.message);
                if (data.status === 'success') {
                    const trainingResult = document.getElementById('trainingResult');
                    trainingResult.innerHTML = `<h5>학습 결과</h5><p>모델 이름: ${data.result.model_name}</p>`;
                }
            });
        });

        loadOptions();
    }
});
// "다음" 버튼 클릭 시 모델 생성 페이지로 이동
const submitSettingsButton = document.getElementById("submitSettings");

if (submitSettingsButton) {
    submitSettingsButton.addEventListener("click", function (event) {
        // 폼 제출 이벤트 방지
        event.preventDefault();

        // 단순히 페이지 이동
        window.location.href = "model_creation.html";
    });
}
