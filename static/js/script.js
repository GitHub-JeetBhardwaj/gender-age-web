// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    const body = document.body;
    const lightBtn = document.getElementById('light-mode');
    const darkBtn = document.getElementById('dark-mode');

    lightBtn.addEventListener('click', () => {
        body.classList.remove('dark-mode');
        fetch('/set_mode/light', { method: 'POST' });
    });

    darkBtn.addEventListener('click', () => {
        body.classList.add('dark-mode');
        fetch('/set_mode/dark', { method: 'POST' });
    });
});

