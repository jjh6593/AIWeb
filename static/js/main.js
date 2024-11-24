document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const dataSettingsForm = document.getElementById('dataSettingsForm');
    const dataPreview = document.getElementById('dataPreview');
    const columnSettings = document.getElementById('columnSettings');
    const targetColumn = document.getElementById('targetColumn');

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
                        loadCSVPreview(file);
                        alert(`${file} 파일을 불러왔습니다.`);
                    });

                    listItem.appendChild(loadButton);
                    serverFileList.appendChild(listItem);
                });

                // 모달 표시
                const modal = new bootstrap.Modal(document.getElementById('loadFileModal'));
                modal.show();
            } else {
                alert(data.message);
            }
        })
        .catch(error => console.error('Error:', error));
    });
    // CSV 데이터 미리보기
    function loadCSVPreview(filename) {
        fetch(`/api/get_csv_data?filename=${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const dataPreview = document.getElementById('dataPreview');
                    const columnSettings = document.getElementById('columnSettings');
                    const targetColumn = document.getElementById('targetColumn');
    
                    // 데이터 미리보기 렌더링
                    const table = document.createElement('table');
                    table.classList.add('table', 'table-striped');
    
                    // 테이블 헤더
                    const thead = document.createElement('thead');
                    const headerRow = document.createElement('tr');
                    data.columns.forEach(col => {
                        const th = document.createElement('th');
                        th.textContent = col;
                        headerRow.appendChild(th);
                    });
                    thead.appendChild(headerRow);
                    table.appendChild(thead);
    
                    // 테이블 바디
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
    
                    // 열 설정 렌더링
                    columnSettings.innerHTML = '';
                    targetColumn.innerHTML = '';
                    let row = null;
                    data.columns.forEach((col, index) => {
                        // 5개 컬럼씩 새로운 row 생성
                        if (index % 6 === 0) {
                            row = document.createElement('div');
                            row.classList.add('row', 'mb-3');
                            columnSettings.appendChild(row);
                        }
    
                        const colDiv = document.createElement('div');
                        colDiv.classList.add('col-md-2');
    
                        colDiv.innerHTML = `
                            <label>${col}</label>
                            <input type="text" class="form-control mb-1" placeholder="단위" name="${col}_unit">
                            <input type="number" class="form-control mb-1" placeholder="최소값" name="${col}_min">
                            <input type="number" class="form-control mb-1" placeholder="최대값" name="${col}_max">
                        `;
                        row.appendChild(colDiv);
    
                        // 타겟 열 옵션 추가
                        const option = document.createElement('option');
                        option.value = col;
                        option.textContent = col;
                        targetColumn.appendChild(option);
                    });
                } else {
                    alert(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
    }
    // function loadCSVPreview(filename) {
    //     fetch(`/api/get_csv_data?filename=${filename}`)
    //     .then(response => response.json())
    //     .then(data => {
    //         if (data.status === 'success') {
    //             // 데이터 미리보기 렌더링
    //             const table = document.createElement('table');
    //             table.classList.add('table', 'table-striped');

    //             // 테이블 헤더
    //             const thead = document.createElement('thead');
    //             const headerRow = document.createElement('tr');
    //             data.columns.forEach(col => {
    //                 const th = document.createElement('th');
    //                 th.textContent = col;
    //                 headerRow.appendChild(th);
    //             });
    //             thead.appendChild(headerRow);
    //             table.appendChild(thead);

    //             // 테이블 바디
    //             const tbody = document.createElement('tbody');
    //             data.data_preview.forEach(row => {
    //                 const tr = document.createElement('tr');
    //                 data.columns.forEach(col => {
    //                     const td = document.createElement('td');
    //                     td.textContent = row[col];
    //                     tr.appendChild(td);
    //                 });
    //                 tbody.appendChild(tr);
    //             });
    //             table.appendChild(tbody);

    //             dataPreview.innerHTML = '';
    //             dataPreview.appendChild(table);

    //             // 열 설정 생성
    //             columnSettings.innerHTML = '';
    //             targetColumn.innerHTML = '';
    //             data.columns.forEach(col => {
    //                 // 단위, 최소, 최대값 입력
    //                 const row = document.createElement('div');
    //                 row.classList.add('mb-3');

    //                 row.innerHTML = `
    //                     <label>${col}</label>
    //                     <input type="text" class="form-control mb-1" placeholder="단위" name="${col}_unit">
    //                     <input type="number" class="form-control mb-1" placeholder="최소값" name="${col}_min">
    //                     <input type="number" class="form-control mb-1" placeholder="최대값" name="${col}_max">
    //                 `;
    //                 columnSettings.appendChild(row);

    //                 // 타겟 열 옵션 추가
    //                 const option = document.createElement('option');
    //                 option.value = col;
    //                 option.textContent = col;
    //                 targetColumn.appendChild(option);
    //             });
    //         } else {
    //             alert(data.message);
    //         }
    //     })
    //     .catch(error => console.error('Error:', error));
    // }

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
});
