// js/result.js

document.addEventListener('DOMContentLoaded', function () {
    // 학습 결과 로드
    function loadResults() {
        fetch('/api/get_training_results')
            .then(response => response.json())
            .then(data => {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = ''; // 기존 내용 초기화
                if (data.results.length > 0) {
                    data.results.forEach(result => {
                        const colDiv = document.createElement('div');
                        colDiv.classList.add('col-md-6');

                        const cardDiv = document.createElement('div');
                        cardDiv.classList.add('card', 'h-100', 'shadow-sm');

                        const cardBodyDiv = document.createElement('div');
                        cardBodyDiv.classList.add('card-body');

                        const title = document.createElement('h5');
                        title.classList.add('card-title');
                        title.textContent = `파일명: ${result.filename}`;

                        const bestConfig = document.createElement('p');
                        bestConfig.classList.add('card-text');
                        bestConfig.textContent = `Best Config: ${JSON.stringify(result.best_config)}`;

                        const bestPred = document.createElement('p');
                        bestPred.classList.add('card-text');
                        bestPred.textContent = `Best Prediction: ${result.best_pred}`;

                        const deleteButton = document.createElement('button');
                        deleteButton.classList.add('btn', 'btn-danger');
                        deleteButton.textContent = '삭제';
                        deleteButton.addEventListener('click', function () {
                            if (confirm('정말로 삭제하시겠습니까?')) {
                                deleteResult(result.filename, colDiv);
                            }
                        });

                        cardBodyDiv.appendChild(title);
                        cardBodyDiv.appendChild(bestConfig);
                        cardBodyDiv.appendChild(bestPred);
                        cardBodyDiv.appendChild(deleteButton);

                        cardDiv.appendChild(cardBodyDiv);
                        colDiv.appendChild(cardDiv);
                        resultsDiv.appendChild(colDiv);
                    });
                } else {
                    resultsDiv.innerHTML = '<p>학습된 결과가 없습니다.</p>';
                }
            })
            .catch(error => console.error('Error:', error));
    }

    function deleteResult(filename, resultDiv) {
        fetch('/api/delete_result', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: filename })
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // 결과 블록 삭제
                    resultDiv.remove();
                } else {
                    alert('삭제에 실패하였습니다.');
                }
            })
            .catch(error => console.error('Error:', error));
    }

    loadResults();
});
