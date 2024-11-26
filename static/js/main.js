document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const dataSettingsForm = document.getElementById('dataSettingsForm');
    const dataPreview = document.getElementById('dataPreview');
    const columnSettings = document.getElementById('columnSettings');
    const targetColumn = document.getElementById('targetColumn');
    const loadFileButton = document.getElementById('loadFileButton');
    const serverFileList = document.getElementById('serverFileList');
    const loadFileModalElement = document.getElementById('loadFileModal');
    const loadFileModal = new bootstrap.Modal(loadFileModalElement);


    let uploadedFilename = '';

    // 파일 업로드
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const fileInput = document.getElementById('fileInput');
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        fetch('/api/upload_csv', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                uploadedFilename = data.filename;
                console.log('Uploaded Filename:', uploadedFilename); // 디버깅용
                alert(data.message);
                loadCSVPreview(data.filename);
            } else {
                alert(data.message);
            }
        })
        .catch(error => console.error('Error:', error));
    });
     // 불러오기 버튼 클릭 이벤트
     loadFileButton.addEventListener('click', function() {

        fetch('/api/get_csv_files')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                serverFileList.innerHTML = '';
                data.files.forEach(file => {
                    const listItem = document.createElement('li');
                    listItem.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
                    listItem.textContent = file;

                    // 불러오기 버튼 추가
                    const loadButton = document.createElement('button');
                    loadButton.textContent = '불러오기';
                    loadButton.classList.add('btn', 'btn-sm', 'btn-primary');
                    loadButton.addEventListener('click', function() {

                        uploadedFilename = file; // 불러온 파일명을 uploadedFilename에 저장
                        console.log('Loaded Filename:', uploadedFilename); // 디버깅용
                        loadCSVPreview(file);
                        alert(`${file} 파일을 불러왔습니다.`);
                        loadFileModal.hide();
                    });

                    listItem.appendChild(loadButton);
                    serverFileList.appendChild(listItem);
                });

                // 모달 표시
                loadFileModal.show();
            } else {
                alert(data.message);
            }
        })
        .catch(error => console.error('Error:', error));
    });

    // 데이터 미리보기와 컬럼 제외 기능 추가
    function loadCSVPreview(filename) {
        fetch(`/api/get_csv_data?filename=${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const dataPreview = document.getElementById('dataPreview');
                    const excludeColumns = document.getElementById('excludeColumns');

                    // 데이터 미리보기 렌더링
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

                    // 컬럼 제외 체크박스 렌더링
                    excludeColumns.innerHTML = '';
                    data.columns.forEach(col => {
                        const div = document.createElement('div');
                        div.classList.add('col-md-3', 'col-sm-6');

                        const formCheckDiv = document.createElement('div');
                        formCheckDiv.classList.add('form-check');

                        const input = document.createElement('input');
                        input.type = 'checkbox';
                        input.classList.add('form-check-input');
                        input.id = `exclude-${col}`;
                        input.value = col;

                        const label = document.createElement('label');
                        label.classList.add('form-check-label');
                        label.setAttribute('for', `exclude-${col}`);
                        label.textContent = col;

                        formCheckDiv.appendChild(input);
                        formCheckDiv.appendChild(label);
                        div.appendChild(formCheckDiv);
                        excludeColumns.appendChild(div);

                    });
                } else {
                    alert(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
    }


    document.getElementById('saveFilteredFile').addEventListener('click', () => {
        const checkedColumns = Array.from(document.querySelectorAll('#excludeColumns input:checked')).map(input => input.value);
        const newFilename = document.getElementById('newFilename').value;

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
            filename: uploadedFilename // 업로드된 또는 불러온 파일 이름을 전달
        };

        fetch('/api/save_filtered_csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert(data.message);
                } else {
                    alert(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
    });

        // 데이터 컬럼 체크박스 동적 생성
    function loadCSVPreview(filename) {
        fetch(`/api/get_csv_data?filename=${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const excludeColumns = document.getElementById('excludeColumns');

                    // 데이터 미리보기 렌더링
                    const dataPreview = document.getElementById('dataPreview');
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

                    // index.html 데이터 설정 - 컬럼 제외 체크박스 렌더링 (가로 배치)
                    excludeColumns.innerHTML = '';
                    data.columns.forEach(col => {
                        const div = document.createElement('div');
                        div.classList.add('col-md-3', 'form-check');

                        const input = document.createElement('input');
                        input.type = 'checkbox';
                        input.classList.add('form-check-input');
                        input.id = `exclude-${col}`;
                        input.value = col;

                        const label = document.createElement('span');
                        label.classList.add('form-check-label');
                        label.setAttribute('for', `exclude-${col}`);
                        label.textContent = col;

                        div.appendChild(input);
                        div.appendChild(label);
                        excludeColumns.appendChild(div);
                    });
                } else {
                    alert(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
    }

    // 데이터 설정 제출
    document.getElementById('submitSettings').addEventListener('click', function() {
        const settings = new FormData(dataSettingsForm);
        settings.append('filename', uploadedFilename);
        settings.append('scaler', document.getElementById('scaler').value);
        settings.append('missingHandling', document.getElementById('missingHandling').value);
        settings.append('targetColumn', document.getElementById('targetColumn').value);

        fetch('/api/submit_data_settings', {
            method: 'POST',
            body: settings
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert(data.message);
                window.location.href = '/model_creation.html';
            } else {
                alert(data.message);
            }
        })
        .catch(error => console.error('Error:', error));
        });
    // 폼 제출 방지 (페이지 새로고침 방지)
    document.getElementById('excludeColumnsForm').addEventListener('submit', function(event) {
        event.preventDefault();
    });

    // 입력 필드에서 Enter 키로 "필터링된 데이터 저장" 버튼 동작
    document.getElementById('newFilename').addEventListener('keydown', function(event) {
        const inputValue = this.value.trim(); // 입력 필드 값 확인
        if (event.key === 'Enter' && inputValue !== '') { // 텍스트가 작성된 경우에만 동작
            event.preventDefault(); // 기본 폼 제출 방지
            document.getElementById('saveFilteredFile').click(); // 버튼 클릭 실행
        }
    });



    });

    document.addEventListener('DOMContentLoaded', function() {
        // 모델 타입별 목록 정의
        const modelOptions = {
            pytorch: ['MLP_1', 'MLP_2', 'MLP_3', 'Conv1D_1'],
            scikit: [
                'LinearRegressor', 'Ridge', 'Lasso', 'ElasticNet',
                'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
                'SVR', 'KNeighborsRegressor', 'HuberRegressor', 'GaussianProcessRegressor', 'XGBoost'
            ]
        };

        // 요소 선택
        const modelTypeSelect = document.getElementById('modelType');
        const modelListDropdown = document.getElementById('modelListDropdown');

        // 모델 타입 선택 시, 모델 목록 표시
        modelTypeSelect.addEventListener('change', function() {
            const selectedType = modelTypeSelect.value;
            console.log('선택된 모델 타입:', selectedType);

            // 모델 목록 초기화
            modelListDropdown.innerHTML = '';

            if (selectedType && modelOptions[selectedType]) {
                console.log('해당 모델 타입의 모델 목록을 표시합니다.');
                // 모델 목록 생성
                modelOptions[selectedType].forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelListDropdown.appendChild(option);
                });

                // 모델 목록 활성화
                modelListDropdown.disabled = false;
            } else {
                console.log('모델 타입이 선택되지 않았거나, 해당하는 모델 목록이 없습니다.');
                // 모델 목록 비활성화
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = '모델 타입을 먼저 선택하세요';
                modelListDropdown.appendChild(defaultOption);
                modelListDropdown.disabled = true;
            }
        });

        // 모델 생성 폼 제출 이벤트
        const modelForm = document.getElementById('modelForm');
        modelForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const modelType = modelTypeSelect.value;
            const modelSelected = modelListDropdown.value;
            const modelName = document.getElementById('modelName').value;

            if (!modelType || !modelSelected || !modelName) {
                alert('모델 타입, 모델 선택, 그리고 모델 이름을 입력하세요.');
                return;
            }

            const modelInfo = {
                model_type: modelType,
                model_name: modelName,
                model_selected: modelSelected
            };

            fetch('/api/save_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(modelInfo)
            })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
                loadModelList();
            })
            .catch(error => console.error('Error:', error));
        });

        // 저장된 모델 목록 로드
        function loadModelList() {
            fetch('/api/get_models')
            .then(response => response.json())
            .then(data => {
                const modelList = document.getElementById('modelList');
                modelList.innerHTML = '';
                data.models.forEach(model => {
                    const li = document.createElement('li');
                    li.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
                    li.textContent = `${model.model_name} (${model.model_selected})`;

                    // 삭제 버튼 추가
                    const deleteButton = document.createElement('button');
                    deleteButton.classList.add('btn', 'btn-danger', 'btn-sm');
                    deleteButton.textContent = '삭제';
                    deleteButton.addEventListener('click', function() {
                        if (confirm(`모델 ${model.model_name}을(를) 삭제하시겠습니까?`)) {
                            fetch('/api/delete_model', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ model_name: model.model_name })
                            })
                            .then(response => response.json())
                            .then(data => {
                                alert(data.message);
                                loadModelList();
                            })
                            .catch(error => console.error('Error:', error));
                        }
                    });

                    li.appendChild(deleteButton);
                    modelList.appendChild(li);
                });
            })
            .catch(error => console.error('Error:', error));
        }

        // 초기 모델 목록 로드
        loadModelList();
    });
    document.addEventListener('DOMContentLoaded', function () {
        // 모델 업로드 폼 제출 이벤트
        const uploadModelForm = document.getElementById('uploadModelForm');
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

            fetch('/api/upload_model', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert(data.message);
                    loadModelList();
                } else {
                    alert(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
        });

        // 기존 로드 모델 목록 로드 함수와 통합 가능
        function loadModelList() {
            fetch('/api/get_models')
            .then(response => response.json())
            .then(data => {
                const modelList = document.getElementById('modelList');
                modelList.innerHTML = '';
                data.models.forEach(model => {
                    const li = document.createElement('li');
                    li.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
                    li.textContent = `${model.model_name} (${model.type || '생성된 모델'})`;

                    // 삭제 버튼 추가
                    const deleteButton = document.createElement('button');
                    deleteButton.classList.add('btn', 'btn-danger', 'btn-sm');
                    deleteButton.textContent = '삭제';
                    deleteButton.addEventListener('click', function () {
                        if (confirm(`모델 ${model.model_name}을(를) 삭제하시겠습니까?`)) {
                            fetch('/api/delete_model', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ model_name: model.model_name })
                            })
                            .then(response => response.json())
                            .then(data => {
                                alert(data.message);
                                loadModelList();
                            })
                            .catch(error => console.error('Error:', error));
                        }
                    });

                    li.appendChild(deleteButton);
                    modelList.appendChild(li);
                });
            })
            .catch(error => console.error('Error:', error));
        }

        // 초기 모델 목록 로드
        loadModelList();
    });