// Reemplaza esto con la URL de tu API
const API_URL = 'http://localhost:5000/predict';

// Cuando el formulario se envía, realiza la petición a la API
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    fetch(API_URL, {
        method: 'POST',
        body: new FormData(event.target)
    }).then(response => response.json()).then(data => {
        const imagesContainer = document.getElementById('images');
        const histogramCanvas = document.getElementById('histogram');
        const histogramCtx = histogramCanvas.getContext('2d');

        // Muestra las imágenes
        data.predictions.forEach((prediction, i) => {
            const img = document.createElement('img');
            img.src = 'data:image/jpeg;base64,' + btoa(prediction.image);
            img.alt = `Imagen ${i + 1}`;
            imagesContainer.appendChild(img);
        });

        // Muestra el histograma
        const histogram = new Chart(histogramCtx, {
            type: 'bar',
            data: {
                labels: data.predictions.map((_, i) => `Imagen ${i + 1}`),
                datasets: [{
                    label: 'Predicción',
                    data: data.predictions.map(prediction => prediction.value),
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                }
            }
        });
    }).catch(error => console.error('Error:', error));
});
