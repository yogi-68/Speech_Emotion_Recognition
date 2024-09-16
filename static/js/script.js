document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.textContent = `Error: ${data.error}`;
            } else {
                resultDiv.textContent = `Predicted emotion: ${data.emotion}`;
            }
        })
        .catch(error => {
            resultDiv.textContent = `Error: ${error}`;
        });
    });
});