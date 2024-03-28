let logInterval; // Store the interval ID to clear it later
function startStream() {
    fetch('/start')  // Send a request to the server to start the stream
        .then(response => response.text())
        .then(data => alert(data));  // Display a success or error message
}

function stopStream() {
    fetch('/stop')  // Send a request to the server to stop the stream
        .then(response => response.text())
        .then(data => {
            alert(data);  // Display a success or error message

            // Display images when stream is stopped
            fetchImages();
        });

}

function fetchLogs() {
    fetch('/logs')
        .then(response => response.text())
        .then(data => {
            const logOutput = document.getElementById('log-output');

            // Check if data is not empty before appending
            if (data.trim() !== '') {
                logOutput.innerHTML += data + '\n';
                logOutput.scrollTop = logOutput.scrollHeight;
            }

            // No need to check for empty content again, update display based on current content
            logOutput.style.display = logOutput.innerText.trim() !== '' ? 'block' : 'none';
        })
        .catch(error => console.error('Error fetching logs:', error));
}

fetchLogs();
if (!logInterval) { // Check if interval is not already running
    logInterval = setInterval(fetchLogs, 2000);
}


function fetchImages() {
    const imageContainer = document.getElementById('image-output');
    imageContainer.innerHTML = ''; // Clear existing images

    fetch('/images') // Assuming a new route to fetch image paths
        .then(response => response.json())  // Parse JSON response (assuming it's a list of image paths)
        .then(data => {
            if (data.length > 0) {
                imageContainer.style.display = 'block';

                for (const imagePath of data) {
                    const imageElement = document.createElement('img');
                    imageElement.src = imagePath;
                    imageContainer.appendChild(imageElement);
                }
            } else {
                imageContainer.style.display = 'none';
            }
        })
        .catch(error => console.error('Error fetching images:', error));
}

const locationForm = document.getElementById('location-form');
locationForm.addEventListener('change', () => {
    const selectedLocation = document.getElementById('location').value;

    if (selectedLocation === '') {
        alert("Please select a Location parameter")
    } else {
        fetch('/set_location', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({location: selectedLocation})
        })
            .then(response => response.text())
            .then(data => alert(data)); // Display success or error message
    }
});

const weatherForm = document.getElementById('weather-form');
weatherForm.addEventListener('change', () => {
    const selectedWeather = document.getElementById('weather').value;

    if (selectedWeather === '') {
        alert("Please select a Weather condition")
    } else {
        fetch('/set_weather', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({weather: selectedWeather})
        })
            .then(response => response.text())
            .then(data => alert(data)); // Display success or error message
    }

});

const modeForm = document.getElementById('mode-form');
modeForm.addEventListener('change', () => {
    const selectedMode = document.getElementById('mode').value;

    fetch('/set_mode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: selectedMode})
    })
        .then(response => response.text())
        .then(data => alert(data)); // Display success or error message
});

document.getElementById('mode').addEventListener('change', function () {
    const mode = this.value;
    const weatherGroup = document.getElementById('weather-group');
    const locationGroup = document.getElementById('location-group');
    const locationSelect = document.getElementById('location');

    switch (mode) {
        case 'Luminance':
            weatherGroup.style.display = 'none';
            locationGroup.style.display = 'none';
            break;
        case 'Weather':
            weatherGroup.style.display = 'block';
            locationGroup.style.display = 'none';
            locationSelect.value = 'Outdoor';
            break;
        case 'Location':
            weatherGroup.style.display = 'none';
            locationGroup.style.display = 'block';
            break;
        default:
            weatherGroup.style.display = 'block';
            locationGroup.style.display = 'block';
    }
});

document.getElementById('location').addEventListener('change', function () {
    const location = this.value;
    const weatherGroup = document.getElementById('weather-group');

    if (location === 'Indoor') {
        weatherGroup.style.display = 'none';
    } else {
        weatherGroup.style.display = 'block';
    }
});