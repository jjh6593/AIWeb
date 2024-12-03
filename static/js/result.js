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
                        title.textContent = `파일명: ${result.folder_name}`;

                        const bestConfig = document.createElement('p');
                        bestConfig.classList.add('card-text');
                        bestConfig.textContent = `Best Config: ${JSON.stringify(result.best_config)}`;

                        const bestPred = document.createElement('p');
                        bestPred.classList.add('card-text');
                        bestPred.textContent = `Best Prediction: ${result.best_pred}`;

                        // 카드 구성 요소 추가
                        cardBodyDiv.appendChild(title);
                        cardBodyDiv.appendChild(bestConfig);
                        cardBodyDiv.appendChild(bestPred);

                        // 그래프를 그릴 canvas 요소 생성
                        const canvas = document.createElement('canvas');
                        const canvasId = `chart-${result.folder_name}`;
                        canvas.id = canvasId;
                        canvas.style.maxWidth = '100%'; // 캔버스 크기 조정
                        cardBodyDiv.appendChild(canvas);

                        // 예측값 처리 및 그래프 그리기
                        const predictions = result.predictions; // 'predictions'로 변경

                        if (typeof predictions === 'number') {
                            // 스칼라 값인 경우
                            const scalarValue = document.createElement('p');
                            scalarValue.textContent = `Prediction: ${predictions}`;
                            cardBodyDiv.appendChild(scalarValue);
                        } else if (Array.isArray(predictions)) {
                            if (predictions.length > 0 && typeof predictions[0] === 'number') {
                                // 1차원 배열인 경우
                                const labels = predictions.map((_, index) => index + 1);
                                const data = predictions;

                                const chartData = {
                                    labels: labels,
                                    datasets: [{
                                        label: 'Prediction',
                                        data: data,
                                        borderColor: 'rgba(75, 192, 192, 1)',
                                        fill: false,
                                    }]
                                };

                                new Chart(canvas, {
                                    type: 'line',
                                    data: chartData,
                                    options: {
                                        scales: {
                                            y: {
                                                beginAtZero: true,
                                            }
                                        },
                                        plugins: {
                                            annotation: {
                                                annotations: {
                                                    line1: {
                                                        type: 'line',
                                                        yMin: 550,
                                                        yMax: 550,
                                                        borderColor: 'rgba(255, 99, 132, 0.5)',
                                                        borderWidth: 2,
                                                        borderDash: [6, 6],
                                                        // label 설정 제거
                                                    }
                                                }
                                            }
                                        }
                                    }
                                });
                            } else if (Array.isArray(predictions[0])) {
                                // 다차원 배열인 경우
                                const numSeries = predictions[0].length;
                                const labels = predictions.map((_, index) => index + 1);

                                const datasets = [];
                                const colors = [
                                    'rgba(75, 192, 192, 1)',
                                    'rgba(255, 99, 132, 1)',
                                    'rgba(54, 162, 235, 1)',
                                    'rgba(255, 206, 86, 1)',
                                    'rgba(153, 102, 255, 1)',
                                    'rgba(255, 159, 64, 1)',
                                    'rgba(201, 203, 207, 1)'
                                ];

                                for (let i = 0; i < numSeries; i++) {
                                    const data = predictions.map(item => item[i]);
                                    datasets.push({
                                        label: `Beam ${i + 1}`,
                                        data: data,
                                        borderColor: colors[i % colors.length],
                                        fill: false,
                                    });
                                }

                                const chartData = {
                                    labels: labels,
                                    datasets: datasets
                                };

                                new Chart(canvas, {
                                    type: 'line',
                                    data: chartData,
                                    options: {
                                        scales: {
                                            y: {
                                                beginAtZero: true,
                                            }
                                        },
                                        plugins: {
                                            annotation: {
                                                annotations: {
                                                    line1: {
                                                        type: 'line',
                                                        yMin: 550,
                                                        yMax: 550,
                                                        borderColor: 'rgba(255, 99, 132, 0.5)',
                                                        borderWidth: 2,
                                                        borderDash: [6, 6],
                                                        // label 설정 제거
                                                    }
                                                }
                                            }
                                        }
                                    }
                                });
                            } else {
                                console.error('예측값의 데이터 형식을 인식할 수 없습니다.');
                            }
                        } else {
                            console.error('예측값의 데이터 타입을 인식할 수 없습니다.');
                        }

                        const deleteButton = document.createElement('button');
                        deleteButton.classList.add('btn', 'btn-danger');
                        deleteButton.textContent = '삭제';
                        deleteButton.addEventListener('click', function () {
                            if (confirm('정말로 삭제하시겠습니까?')) {
                                deleteResult(result.filename, colDiv);
                            }
                        });

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
                    resultDiv.remove();
                } else {
                    alert('삭제에 실패하였습니다.');
                }
            })
            .catch(error => console.error('Error:', error));
    }

    loadResults();
});
