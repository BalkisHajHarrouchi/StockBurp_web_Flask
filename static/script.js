document.addEventListener("DOMContentLoaded", function () {
    const containernew = document.getElementById('containernew');
    const registerBtn = document.getElementById('register');
    const loginBtn = document.getElementById('login');

    registerBtn.addEventListener('click', () => {
        containernew.classList.add("active");
    });

    loginBtn.addEventListener('click', () => {
        containernew.classList.remove("active");
    });
});
