document.addEventListener('DOMContentLoaded', function() {
    const pageSequence = ["index.html", "model_creation.html", "model_training.html", "input_prediction.html", "training_results.html"];
    const nextButton = document.getElementById("nextButton");

    if (nextButton) {
        console.log("nextButton 이벤트 등록");
        nextButton.addEventListener("click", function (event) {
            event.preventDefault();
            
            // 현재 페이지 URL에서 파일 이름 추출
            let currentPage = window.location.pathname.split("/").pop();
            console.log(currentPage)
            // 루트 경로 ('/')를 index.html로 간주
            if (!currentPage || currentPage == "") {
                currentPage = "index.html";
                
            }
            // 페이지 인덱스 계산
            const currentIndex = pageSequence.indexOf(currentPage);

            // 현재 페이지가 유효하고 다음 페이지가 있는 경우에만 이동
            if (currentIndex !== -1 && currentIndex < pageSequence.length - 1) {
                const nextPage = pageSequence[currentIndex + 1];
                window.location.href = nextPage;
            } else {
                alert("더 이상 이동할 페이지가 없습니다.");
            }
        });
    }
});
