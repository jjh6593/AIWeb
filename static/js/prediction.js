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
    const uploadedFilename = localStorage.getItem('uploadedFilename');
    if (!uploadedFilename) {
        alert('CSV 파일이 선택되지 않았습니다.');
        // 필요에 따라 이전 페이지로 리다이렉트할 수 있습니다.
        return;
    }

    // 초기 설정
    toleranceInput.style.display = 'none';
    startingPointInput.style.display = 'none';

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
        title.textContent = 'Starting Point';
        title.className = 'mt-3 mb-3';
        startingPointInput.appendChild(title);

        const row = document.createElement('div'); // Bootstrap Row 생성
        row.className = 'row';
        startingPointInput.appendChild(row); // Row 추가

        columns.forEach(col => {
            const colWrapper = document.createElement('div'); // Wrapper (Column)
            colWrapper.className = 'col-md-2 mb-3'; // Grid 레이아웃 (3열 구조)
            row.appendChild(colWrapper); // Row에 추가

            // input field를 Wrapper 안에 생성
            createInputField(col, `startingPoint_${col}`, `startingPoint_${col}`, 'text', colWrapper);
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

    // Modeling List 선택에 따른 Strategy 업데이트
    modelingListDropdown.addEventListener('change', function () {
        const selectedMode = modeTypeSelect.value.toLowerCase();
        const selectedModeling = modelingListDropdown.value;

        resetDropdown(strategyListDropdown, '전략 타입을 선택하세요');

        if (options[selectedMode] && options[selectedMode][selectedModeling]) {
            const strategyOptions = options[selectedMode][selectedModeling];
            populateDropdown(strategyListDropdown, strategyOptions, '전략 타입을 선택하세요');
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
});
