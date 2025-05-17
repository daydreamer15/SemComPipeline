# ESemCom Demo

This is a simple GUI demonstration of the Evolving Semantic Communication (ESemCom) system described in the paper "Evolving Semantic Communication with Generative Model". The application allows users to upload an image and see how it's processed through the various stages of the ESemCom system.

## Project Structure

The project consists of two main components:

1. **Backend**: A Flask application that handles image processing and simulates the ESemCom pipeline.
2. **Frontend**: A Vue.js application that provides a user-friendly interface for uploading images and viewing the results.

## Setup and Installation

### Prerequisites

- Python 3.7+
- Node.js 14+
- npm 6+

### Backend Setup

1. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required Python packages:

   ```
   pip install flask flask-cors opencv-python numpy
   ```

3. Run the Flask application:
   ```
   python app.py
   ```

### Frontend Setup

1. Install the required Node.js packages:

   ```
   npm install
   ```

2. Run the Vue.js development server:

   ```
   npm run serve
   ```

3. Build for production:
   ```
   npm run build
   ```

## Usage

1. Open your browser and navigate to `http://localhost:8080/` (or the URL provided by the Vue.js development server).
2. Click on the "Choose Image" button to select an image from your computer.
3. Adjust the Channel Quality (SNR) slider if desired.
4. Click on the "Process Image" button to see how the image is processed through the ESemCom pipeline.
5. View the results, including the different stages of processing and performance metrics.

## Simulated Pipeline Stages

1. **Original Image**: The input image uploaded by the user.
2. **Semantic Encoder**: Simulates the Channel-aware Semantic Encoder (StyleGAN inversion).
3. **Transmitter Cache**: Simulates the Semantic Caching at the transmitter.
4. **Power Normalization**: Simulates the power normalization process.
5. **Noisy Channel**: Simulates the AWGN channel by adding Gaussian noise.
6. **Receiver Cache**: Simulates the Semantic Caching at the receiver.
7. **Semantic Decoder**: Simulates the Semantic Decoder (StyleGAN generator) that reconstructs the image.

## Performance Metrics

The application displays the following simulated performance metrics:

- **BCR (Bandwidth Compression Ratio)**: The ratio of the number of transmitted symbols to the number of source symbols.
- **PSNR (Peak Signal-to-Noise Ratio)**: A measure of the quality of the reconstructed image compared to the original.
- **LPIPS (Learned Perceptual Image Similarity)**: A perceptual metric that measures the similarity between two images.

## Notes

- This is a simplified simulation of the ESemCom system described in the paper.
- The actual implementation would require a trained StyleGAN model and more complex processing.
