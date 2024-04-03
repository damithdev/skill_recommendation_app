document.querySelector('.browse-link').addEventListener('click', function () {
    document.querySelector('input[type=file]').click();
});

document.querySelector('input[type=file]').addEventListener('change', handleFileUpload);

function handleFileUpload(event) {
    let file = event.target.files[0];
    if (!file) {
        return;
    }
    uploadFile(file);
}

function uploadFile(file) {
    let formData = new FormData();
    formData.append('file', file);
    // Show loading indicator
    document.getElementById('loading-indicator').style.display = 'block';


    setTimeout(() => {
        fetch('/', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || 'Server responded with an error!');
                    });
                }
                return response.json();
            })
            .then(careers => {
                updateCareersSection(careers);
                // Hide loading indicator
                document.getElementById('loading-indicator').style.display = 'none';
            })
            .catch(error => {
                console.error('Error:', error);
                showErrorToast(error.message);

                document.getElementById('loading-indicator').style.display = 'none';
            });
    }, 3000);

}

function showErrorToast(message) {
    const toastEl = document.getElementById('error-toast');
    const toastBody = toastEl.querySelector('.toast-body');
    toastBody.textContent = message;

    const bootstrapToast = new bootstrap.Toast(toastEl);
    bootstrapToast.show();
}


const dragDropArea = document.querySelector('.upload-box');

dragDropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dragDropArea.classList.add('dragover-style');

});
dragDropArea.addEventListener('dragleave', (event) => {
    event.preventDefault();
    dragDropArea.classList.remove('dragover-style');
});
dragDropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDropArea.classList.remove('dragover-style'); // Remove the style on drop as well

    let file = event.dataTransfer.files[0];
    uploadFile(file);
});

// Get the back button
const backButton = document.querySelector('#back-icon');
const detailsButton = document.querySelector('#details-btn');

const uploadSection = document.querySelector('#upload-section');
const careersSection = document.querySelector('#careers-section');
const detailsSection = document.querySelector('#details-section');
// Add click event listener to the back button
backButton.addEventListener('click', function () {
    // Toggle visibility of the upload box and details section
    uploadSection.style.display = 'block';
    detailsSection.style.display = 'none';
    document.querySelector('#back-icon-div').style.display = 'none';
    document.querySelectorAll('.recommendation-card').forEach(card => card.classList.remove('selected-card'));
    document.querySelectorAll('.details-btn').forEach(button => button.disabled = false);

});

function updateCareersSection(careers) {
    let careersHtml = '';

    careers.forEach((career, index) => {
        careersHtml += `
            <div class="recommendation-card">
                <h6>${career.job}</h6>
                <button class="btn btn-primary details-btn" data-index="${index}">Details</button>
            </div>
        `;
    });

    document.getElementById('careers-section').innerHTML = careersHtml;

    // Add event listeners to the new "Details" buttons
    document.querySelectorAll('.details-btn').forEach(button => {
        button.addEventListener('click', function () {
            const index = this.getAttribute('data-index');
            const selectedCareer = careers[index];
            // Highlight the selected card
            document.querySelectorAll('.recommendation-card').forEach(card => card.classList.remove('selected-card'));
            this.closest('.recommendation-card').classList.add('selected-card');

            populateDetailsSection(selectedCareer);
            this.disabled = true;

        });
    });
}

function populateDetailsSection(career) {

    document.querySelector('#details-section .text-primary').textContent = `${career.job}`;

    let skillsHtml = career.skills.map(skill => `<li>${skill}</li>`).join('');

    // Assume details-section has a <ul> for skills
    document.querySelector('#details-section ul').innerHTML = skillsHtml;

    // Show the details section and hide other sections if necessary
    document.querySelector('#back-icon-div').style.display = 'block';
    document.querySelector('#details-section').style.display = 'block';
    document.querySelector('#upload-section').style.display = 'none';
    document.querySelectorAll('.details-btn').forEach(button => button.disabled = false);

}