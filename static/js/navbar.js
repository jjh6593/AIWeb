document.addEventListener('DOMContentLoaded', function () {
    const navbarContainer = document.getElementById('navbar');

    // 현재 페이지 URL에서 파일 이름 추출
    let currentPage = window.location.pathname.split('/').pop().replace('.html', '');

    // 만약 currentPage가 빈 문자열이라면, 'index'로 설정
    if (currentPage === '') {
        currentPage = 'index';
    }

    // 네비게이션 바 HTML 파일 로드
    fetch('navbar.html')
        .then(response => response.text())
        .then(data => {
            // 네비게이션 바 삽입
            navbarContainer.innerHTML = data;

            // 현재 페이지에 해당하는 링크에 active 클래스 추가
            const activeLink = document.querySelector(`a.nav-link[data-page="${currentPage}"]`);
            if (activeLink) {
                activeLink.classList.add('active');
            }
        })
        .catch(error => console.error('네비게이션 바 로드 실패:', error));
});
