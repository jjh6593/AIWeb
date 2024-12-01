document.addEventListener("DOMContentLoaded", function () {
    const options = {
        global: {
            Single: ['Beam', 'Stochastic', 'Best One'],
            Averaging: ['Beam', 'Stochastic', 'Best One'],
            Ensemble: ['Beam', 'Stochastic', 'Best One']
        },
        local: {
            Single: ['Brute Force', 'Manual', 'Sensitivity'],
            Averaging: ['Brute Force', 'Manual', 'Sensitivity'],
            Ensemble: ['Brute Force', 'Manual', 'Sensitivity']
        }
    };

    // 공통 함수 정의
    function fetchJSON(url, options = {}, callback) {
        fetch(url, options)
            .then(response => response.json())
            .then(data => callback(data))
            .catch(error => console.error('Error:', error));
    }

    // URL에서 쿼리 파라미터 추출 함수
    function getQueryParam(param) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(param);
    }

    // DOM 요소 선택
    const predictionForm = document.getElementById('predictionForm');
    const modeTypeSelect = document.getElementById('modeType');
    const modelingListDropdown = document.getElementById('modelingListDropdown');
    const strategyListDropdown = document.getElementById('strategyListDropdown');
    const toleranceInput = document.getElementById('toleranceInput');
    const startingPointInput = document.getElementById('startingPointInput');
    const additionalInputsContainer = document.getElementById('additionalInputsContainer');
    const modelSelectionContainer = document.getElementById('modelSelectionContainer'); // 모델 선택 컨테이너 추가
    const uploadedFilename = localStorage.getItem('uploadedFilename');

    // 선택된 모델들을 저장할 배열
    let selectedModels = [];

    if (!uploadedFilename) {
        alert('CSV 파일이 선택되지 않았습니다.');
        // 필요에 따라 이전 페이지로 리다이렉트할 수 있습니다.
        return;
    }

    // 초기 설정
    toleranceInput.style.display = 'none';
    startingPointInput.style.display = 'none';
    modelSelectionContainer.style.display = 'none'; // 모델 선택 컨테이너 숨김

    // CSV 데이터 로드 및 startingPointInput 생성
    function loadCSVPreview(filename) {
        fetchJSON(`/api/get_csv_data?filename=${filename}`, {}, data => {
            if (data.status === 'success') {
                renderStartingPointInputs(data.columns);
            } else {
                alert(data.message);
            }
        });
    }

    function renderStartingPointInputs(columns) {
        startingPointInput.innerHTML = ''; // 기존 입력 필드 초기화

        const title = document.createElement('h5');
        title.textContent = 'Starting Point / Unit';
        title.className = 'mt-3 mb-3';
        startingPointInput.appendChild(title);

        const row = document.createElement('div'); // Bootstrap Row 생성
        row.className = 'row';
        startingPointInput.appendChild(row); // Row 추가

        const defaultUnits = [5, 5, 10, 0.5, 5, 5, 5, 5, 5, 60, 0.05, 1, 5, 60, 1000];
        const defaultStartingPoints = [150, 25, 40, 1, 120, 250, 10, 25, 25, 900, 0.25, 2, 100, 1800, 2000];

        let defaultIndex = 0; // 기본값 배열 인덱스
        columns.forEach(col => {
            if (col !== 'Target' && col !== 'target') {
                const colWrapper = document.createElement('div'); // Wrapper (Column)
                colWrapper.className = 'col-md-3 mb-3'; // Grid 레이아웃
                row.appendChild(colWrapper); // Row에 추가

                const label = document.createElement('label');
                label.textContent = `${col} / ${col}_unit`;
                label.className = 'form-label fw-bold';
                colWrapper.appendChild(label);

                // Starting Point 입력 필드 생성
                const startingPointInput = document.createElement('input');
                startingPointInput.type = 'text';
                startingPointInput.name = `startingPoint_${col}`;
                startingPointInput.id = `startingPoint_${col}`;
                startingPointInput.className = 'form-control mb-2';
                startingPointInput.value = defaultStartingPoints[defaultIndex] || '';
                colWrapper.appendChild(startingPointInput);

                // Unit 입력 필드 생성
                const unitInput = document.createElement('input');
                unitInput.type = 'text';
                unitInput.name = `unit_${col}`;
                unitInput.id = `unit_${col}`;
                unitInput.className = 'form-control';
                unitInput.value = defaultUnits[defaultIndex] || '';
                colWrapper.appendChild(unitInput);

                defaultIndex++;
            }
        });
    }

    // 공통 함수들
    function resetDropdown(dropdown, placeholder) {
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        dropdown.disabled = true;
    }

    function populateDropdown(dropdown, optionsArray, placeholder) {
        resetDropdown(dropdown, placeholder);
        optionsArray.forEach(optionValue => {
            const option = document.createElement('option');
            option.value = optionValue;
            option.textContent = optionValue;
            dropdown.appendChild(option);
        });
        dropdown.disabled = false;
    }

    function createInputField(labelText, inputName, inputId, inputType = 'text', parentElement = additionalInputsContainer) {
        const label = document.createElement('label');
        label.htmlFor = inputId;
        label.textContent = labelText;
        label.className = 'form-label fw-bold';

        const input = document.createElement('input');
        input.type = inputType;
        input.name = inputName;
        input.id = inputId;
        input.className = 'form-control mb-3';

        parentElement.appendChild(label);
        parentElement.appendChild(input);
    }

    function createRadioButtonGroup(groupName, options) {
        const divContainer = document.createElement('div');
        divContainer.className = 'mb-3';

        const groupLabel = document.createElement('label');
        groupLabel.textContent = groupName.charAt(0).toUpperCase() + groupName.slice(1);
        groupLabel.className = 'form-label fw-bold';
        divContainer.appendChild(groupLabel);

        options.forEach(option => {
            const div = document.createElement('div');
            div.className = 'form-check';

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = groupName;
            input.value = option.value;
            input.id = `${groupName}_${option.value}`;
            input.className = 'form-check-input';

            const label = document.createElement('label');
            label.className = 'form-check-label';
            label.htmlFor = input.id;
            label.textContent = option.label;

            div.appendChild(input);
            div.appendChild(label);
            divContainer.appendChild(div);
        });

        additionalInputsContainer.appendChild(divContainer);
    }

    // Mode Type 선택에 따른 드롭다운 업데이트
    modeTypeSelect.addEventListener('change', function () {
        const selectedMode = modeTypeSelect.value.toLowerCase();

        resetDropdown(modelingListDropdown, '모델링 타입을 선택하세요');
        resetDropdown(strategyListDropdown, '전략 타입을 선택하세요');
        modelSelectionContainer.style.display = 'none'; // 모델 선택 컨테이너 숨김

        if (options[selectedMode]) {
            const modelingOptions = Object.keys(options[selectedMode]);
            populateDropdown(modelingListDropdown, modelingOptions, '모델링 타입을 선택하세요');

            // Tolerance 입력 표시 설정
            toleranceInput.style.display = selectedMode === 'global' ? 'block' : 'none';
            loadCSVPreview(uploadedFilename);
            startingPointInput.style.display = 'block';
        } else {
            toleranceInput.style.display = 'none';
        }
    });

    // Modeling List 선택에 따른 Strategy 업데이트 및 모델 선택 UI 생성
    modelingListDropdown.addEventListener('change', function () {
        const selectedMode = modeTypeSelect.value.toLowerCase();
        const selectedModeling = modelingListDropdown.value;

        resetDropdown(strategyListDropdown, '전략 타입을 선택하세요');
        modelSelectionContainer.style.display = 'none'; // 모델 선택 컨테이너 숨김

        if (options[selectedMode] && options[selectedMode][selectedModeling]) {
            const strategyOptions = options[selectedMode][selectedModeling];
            populateDropdown(strategyListDropdown, strategyOptions, '전략 타입을 선택하세요');

            // 모델 선택 UI 생성
            if (['Single', 'Averaging', 'Ensemble'].includes(selectedModeling)) {
                loadModels(selectedModeling);
            }
        }
    });

    // Strategy 선택에 따른 추가 입력 필드 생성
    strategyListDropdown.addEventListener('change', function () {
        const selectedStrategy = strategyListDropdown.value;
        additionalInputsContainer.innerHTML = ''; // 기존 입력 필드 초기화

        switch (selectedStrategy) {
            case 'Beam':
                createInputField('Beam width', 'beamWidth', 'beamWidth');
                break;
            case 'Stochastic':
                createInputField('Number of candidates', 'numCandidates', 'numCandidates');
                break;
            case 'Best One':
                createRadioButtonGroup('escape', [
                    { value: 'true', label: 'True' },
                    { value: 'false', label: 'False' }
                ]);
                break;
            case 'Brute Force':
                createInputField('Top K', 'topK', 'topK');
                createRadioButtonGroup('partialKeep', [
                    { value: 'true', label: 'True' },
                    { value: 'false', label: 'False' }
                ]);
                break;
            case 'Manual':
                createInputField('Index', 'index', 'index');
                createRadioButtonGroup('up', [
                    { value: 'true', label: 'True' },
                    { value: 'false', label: 'False' }
                ]);
                break;
            // 추가 전략이 있을 경우 여기에 추가
            default:
                break;
        }
    });

    // 추가된 코드 시작 ------------------------------------------------------------------

    // 모델 목록 로드 및 모델 선택 UI 생성
    function loadModels(selectedModeling) {
        fetchJSON('/api/get_models', {}, data => {
            if (data.status === 'success' && data.models.length > 0) {
                renderModelSelection(data.models, selectedModeling);
            } else {
                alert('저장된 모델이 없습니다.');
                modelSelectionContainer.style.display = 'none';
            }
        });
    }

    // 모델 선택 UI 생성
    function renderModelSelection(models, selectedModeling) {
        modelSelectionContainer.innerHTML = ''; // 기존 UI 초기화
        modelSelectionContainer.style.display = 'block'; // 모델 선택 컨테이너 표시

        if (selectedModeling === 'Single') {
            // Single 모드: 드롭다운으로 모델 선택
            const label = document.createElement('label');
            label.textContent = '모델을 선택하세요';
            label.className = 'form-label';

            const select = document.createElement('select');
            select.className = 'form-select';
            select.id = 'modelSelect';
            select.name = 'model_selected';

            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_name;
                option.textContent = model.model_name;
                select.appendChild(option);
            });

            modelSelectionContainer.appendChild(label);
            modelSelectionContainer.appendChild(select);

            // 모델 선택 변경 시 선택된 모델 업데이트
            select.addEventListener('change', function () {
                selectedModels = [select.value];
            });

            // 초기 선택값 설정
            if (models.length > 0) {
                selectedModels = [models[0].model_name];
            }
        } else if (selectedModeling === 'Averaging' || selectedModeling === 'Ensemble') {
            // Averaging 또는 Ensemble 모드: 체크박스로 모델 선택
            const label = document.createElement('label');
            label.textContent = '모델들을 선택하세요';
            label.className = 'form-label';

            modelSelectionContainer.appendChild(label);

            models.forEach(model => {
                const div = document.createElement('div');
                div.className = 'form-check';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.className = 'form-check-input';
                checkbox.value = model.model_name;
                checkbox.id = `model_${model.model_name}`;
                checkbox.name = 'models_selected[]'; // 폼 제출 시 배열로 전송

                const label = document.createElement('label');
                label.className = 'form-check-label';
                label.htmlFor = checkbox.id;
                label.textContent = model.model_name;

                div.appendChild(checkbox);
                div.appendChild(label);
                modelSelectionContainer.appendChild(div);

                // 체크박스 상태 변경 시 선택된 모델 업데이트
                checkbox.addEventListener('change', function () {
                    if (checkbox.checked) {
                        if (!selectedModels.includes(checkbox.value)) {
                            selectedModels.push(checkbox.value);
                        }
                    } else {
                        selectedModels = selectedModels.filter(name => name !== checkbox.value);
                    }
                });
            });
        } else {
            // 모델 선택이 필요 없는 경우
            modelSelectionContainer.style.display = 'none';
            selectedModels = [];
        }
    }

    // 추가된 코드 끝 ------------------------------------------------------------------

    // 아래 코드 추가 ------------------------------------------------------------------

    // 폼 제출 이벤트 처리
    predictionForm.addEventListener('submit', function (event) {
        event.preventDefault(); // 폼의 기본 동작 막기

        // 데이터 수집 객체 생성
        const data = {};

        // 사용하는 데이터의 주소
        data.filename = uploadedFilename;

        // 사용자가 입력한 Desire
        data.desire = document.getElementById('desire').value;

        // Option 값 (Global 또는 Local)
        data.option = modeTypeSelect.value.toLowerCase();

        // Modeling 타입에 선택된 값
        data.modeling_type = modelingListDropdown.value;

        // Strategy 선택된 값
        data.strategy = strategyListDropdown.value;

        // Global이면 Tolerance 추가
        if (data.option === 'global') {
            data.tolerance = document.getElementById('tolerance').value;
        }

        // Starting Point list와 unit 리스트 수집
        data.starting_points = {};
        data.units = {};

        const startingPointInputs = document.querySelectorAll('[id^="startingPoint_"]');
        startingPointInputs.forEach(input => {
            const colName = input.id.replace('startingPoint_', '');
            data.starting_points[colName] = input.value;
        });

        const unitInputs = document.querySelectorAll('[id^="unit_"]');
        unitInputs.forEach(input => {
            const colName = input.id.replace('unit_', '');
            data.units[colName] = input.value;
        });

        // 사용하는 모델들의 model_name 값 수집
        data.models = selectedModels;

        // 추가 정보 수집 (전략에 따라)
        if (data.option === 'global') {
            switch (data.strategy) {
                case 'Beam':
                    data.beam_width = document.getElementById('beamWidth').value;
                    break;
                case 'Stochastic':
                    data.num_candidates = document.getElementById('numCandidates').value;
                    break;
                case 'Best One':
                    data.escape = document.querySelector('input[name="escape"]:checked')?.value;
                    break;
                // 필요에 따라 추가 전략 처리
            }
        } else if (data.option === 'local') {
            switch (data.strategy) {
                case 'Brute Force':
                    data.top_k = document.getElementById('topK').value;
                    data.partial_keep = document.querySelector('input[name="partialKeep"]:checked')?.value;
                    break;
                case 'Manual':
                    data.index = document.getElementById('index').value;
                    data.up = document.querySelector('input[name="up"]:checked')?.value;
                    break;
                case 'Sensitivity':
                    // 추가 정보 없음
                    break;
                // 필요에 따라 추가 전략 처리
            }
        }

        // 데이터 전송
        fetch('/api/submit_prediction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        }).then(response => {
            if (!response.ok) {
                throw new Error('서버 응답이 올바르지 않습니다.');
            }
            return response.json();
        }).then(result => {
            if (result.status === 'success') {
                // 성공 시 처리 로직
                alert('예측이 완료되었습니다.');
                // 필요에 따라 다음 단계로 이동하거나 결과를 표시
            } else {
                alert('에러: ' + result.message);
            }
        }).catch(error => {
            console.error('Error:', error);
            alert('예측 중 오류가 발생했습니다.');
        });
    });

    // 아래 코드 추가 끝 ---------------------------------------------------------------

});
