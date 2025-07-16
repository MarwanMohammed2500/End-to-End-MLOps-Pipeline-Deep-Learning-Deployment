// FashionMNIST Predictor JavaScript

class FashionMNISTPredictor {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.currentFile = null;
        this.apiBaseUrl = this.getApiBaseUrl();
    }

    getApiBaseUrl() {
        // For Flask applications, use relative paths
        return '';
    }

    initializeElements() {
        // Get DOM elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.selectButton = document.getElementById('selectButton');
        this.predictButton = document.getElementById('predictButton');
        this.previewSection = document.getElementById('previewSection');
        this.imagePreview = document.getElementById('imagePreview');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultValue = document.getElementById('resultValue');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.errorMessage = document.getElementById('errorMessage');
        this.successMessage = document.getElementById('successMessage');
    }

    bindEvents() {
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files?.[0];
            if (file && file !== this.currentFile) {
                this.handleFileSelect(file);
            }
        });

        // Select button click
        this.selectButton.addEventListener('click', () => {
            this.fileInput.click();
        });

        // Predict button click
        this.predictButton.addEventListener('click', () => {
            this.makePrediction();
        });

        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG, GIF, or BMP).');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size must be less than 10MB.');
            return;
        }

        this.currentFile = file;
        this.displayPreview(file);
        this.hideMessages();
    }

    displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.imagePreview.src = e.target.result;
            this.previewSection.classList.add('show');
            this.predictButton.disabled = false;
            this.predictButton.textContent = 'Predict Fashion Item';
        };
        reader.readAsDataURL(file);
    }

    async makePrediction() {
        if (!this.currentFile) {
            this.showError('Please select an image first.');
            return;
        }

        this.showLoading();
        this.hideMessages();
        this.predictButton.disabled = true;

        try {
            // Create FormData
            const formData = new FormData();
            formData.append('file', this.currentFile);

            // Make API call
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Handle different response formats
            let prediction;
            if (data.response) {
                prediction = data.response;
            } else if (data.prediction) {
                prediction = data.prediction;
            } else if (data.result) {
                prediction = data.result;
            } else {
                prediction = JSON.stringify(data);
            }

            this.displayResult(prediction);
            this.showSuccess('Prediction completed successfully!');

        } catch (error) {
            console.error('Prediction error:', error);
            
            let errorMessage = 'Prediction failed. ';
            if (error.message.includes('Failed to fetch')) {
                errorMessage += 'Please check if the backend server is running.';
            } else if (error.message.includes('HTTP error')) {
                errorMessage += `Server error: ${error.message}`;
            } else {
                errorMessage += error.message;
            }
            
            this.showError(errorMessage);
        } finally {
            this.hideLoading();
            this.predictButton.disabled = false;
        }
    }

    displayResult(prediction) {
        this.resultValue.textContent = prediction;
        this.resultsSection.classList.add('show');
        this.resultsSection.classList.add('fade-in');
    }

    showLoading() {
        this.loadingSpinner.classList.add('show');
        this.resultsSection.classList.remove('show');
    }

    hideLoading() {
        this.loadingSpinner.classList.remove('show');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.classList.add('show');
        this.successMessage.classList.remove('show');
    }

    showSuccess(message) {
        this.successMessage.textContent = message;
        this.successMessage.classList.add('show');
        this.errorMessage.classList.remove('show');
    }

    hideMessages() {
        this.errorMessage.classList.remove('show');
        this.successMessage.classList.remove('show');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FashionMNISTPredictor();
});

// Utility functions for better UX
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Add keyboard support
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        const activeElement = document.activeElement;
        if (activeElement && activeElement.classList.contains('upload-area')) {
            e.preventDefault();
            activeElement.click();
        }
    }
});

// Add accessibility support
document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.setAttribute('role', 'button');
        uploadArea.setAttribute('aria-label', 'Click to select image file or drag and drop');
        uploadArea.setAttribute('tabindex', '0');
    }
});