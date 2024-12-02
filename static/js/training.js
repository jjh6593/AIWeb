document.addEventListener('DOMContentLoaded', function () {
    const modelList = document.getElementById('modelList'); // 모델 목록을 표시할 <ul> 요소
    const modelListDropdown = document.getElementById('modelListDropdown'); // 모델 선택 드롭다운 (선택적)

    // 모델 목록을 가져와 렌더링하는 함수
    function loadModelList() {
        if (modelList) {
            fetchModels(models => {
                modelList.innerHTML = ''; // 기존 목록 초기화
                models.forEach(model => {
                    const li = document.createElement('li');
                    li.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
                    // li.textContent = `${model.model_name} (${model.model_selected})`; // framework 정보 표시
                    // 모델 이름 및 framework 정보 표시
                    li.innerHTML = `
                        <span>
                            ${model.model_name} (${model.model_selected})
                            <br>
                            <small>Train Loss: ${model.train_loss.toFixed(4)},   Val Loss: ${model.val_loss.toFixed(4)}</small>

                        </span>
                    `;

                    // 삭제 버튼 추가
                    const deleteButton = document.createElement('button');
                    deleteButton.classList.add('btn', 'btn-danger', 'btn-sm');
                    deleteButton.textContent = '삭제';
                    deleteButton.addEventListener('click', function () {
                        if (confirm(`모델 ${model.model_name}을(를) 삭제하시겠습니까?`)) {
                            deleteModel(model.model_name);
                        }
                    });

                    li.appendChild(deleteButton); // 목록 항목에 삭제 버튼 추가
                    modelList.appendChild(li); // 목록에 항목 추가
                });

                // 선택적: 드롭다운 업데이트
                if (modelListDropdown) {
                    updateModelDropdown(models);
                }
            });
        }
    }

    // 모델 목록 가져오는 API 호출
    function fetchModels(callback) {
        fetch('/api/get_models')
            .then(response => response.json())
            .then(data => {
                if (data.models) {
                    callback(data.models);
                } else {
                    alert('모델 목록을 불러오지 못했습니다.');
                }
            })
            .catch(error => console.error('Error fetching model list:', error));
    }

    // 모델 삭제 함수
    function deleteModel(modelName) {
        fetch('/api/delete_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert(`모델 ${modelName}이(가) 삭제되었습니다.`);
                    loadModelList(); // 목록 다시 로드
                } else {
                    alert(`모델 삭제 중 오류 발생: ${data.message}`);
                }
            })
            .catch(error => console.error('Error deleting model:', error));
    }

    // 선택적: 드롭다운 업데이트 함수
    function updateModelDropdown(models) {
        modelListDropdown.innerHTML = ''; // 기존 드롭다운 초기화
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_name;
            option.textContent = model.model_name;
            modelListDropdown.appendChild(option);
        });
    }

    // 페이지 로드 시 모델 목록 로드
    loadModelList();
});
