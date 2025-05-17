import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import time
import argparse
from PIL import Image

# Import the CelebAMask-HQ model architecture
# Note: Actual implementation would need to download the model weights
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class FaceParsing(nn.Module):
    """
    Simplified version of the face parsing model from CelebAMask-HQ
    """
    def __init__(self, num_classes=19):
        super(FaceParsing, self).__init__()
        self.encoder1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)
        
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)

class SemanticEncoder(nn.Module):
    """
    Semantic encoder that compresses face parsing results into a low-dimensional representation
    """
    def __init__(self, num_classes=19, latent_dim=256):
        super(SemanticEncoder, self).__init__()
        # Use the face parsing model as a feature extractor
        self.face_parsing = FaceParsing(num_classes)
        
        # Add compression layers
        self.compress = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Determine the size after compression
        self.fc_size = self._get_fc_size(256, 256)
        
        # Fully connected layers for encoding
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_size, latent_dim),
            nn.Tanh()  # Bound outputs to [-1, 1] for transmission
        )
    
    def _get_fc_size(self, h, w):
        # Calculate output size after compression
        h_out = h // 8  # After 3 max pooling layers
        w_out = w // 8
        return 16 * h_out * w_out
    
    def forward(self, x):
        # Get semantic segmentation
        semantic_maps = self.face_parsing(x)
        
        # Compress the semantic maps
        compressed = self.compress(semantic_maps)
        
        # Encode to latent representation
        latent = self.fc(compressed)
        
        return semantic_maps, latent

class SemanticDecoder(nn.Module):
    """
    Semantic decoder that reconstructs the image from the latent representation
    """
    def __init__(self, latent_dim=256, num_classes=19):
        super(SemanticDecoder, self).__init__()
        self.num_classes = num_classes
        
        # Size of feature maps after decompression
        self.h_out = 32  # 256 // 8
        self.w_out = 32  # 256 // 8
        
        # FC layer to decompress
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16 * self.h_out * self.w_out),
            nn.ReLU(inplace=True)
        )
        
        # Reshape and upsample
        self.decompress = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )
        
        # Image reconstruction from segmentation masks
        self.reconstruct = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output RGB image with values [0, 1]
        )
    
    def forward(self, latent):
        # Decompress from latent space
        fc_out = self.fc(latent)
        fc_out = fc_out.view(-1, 16, self.h_out, self.w_out)
        
        # Decompress to semantic segmentation
        semantic_maps = self.decompress(fc_out)
        
        # Reconstruct the image from semantic maps
        reconstructed = self.reconstruct(semantic_maps)
        
        return semantic_maps, reconstructed

# AWGN Channel Simulation
class NoisyChannel:
    def __init__(self, snr_db=20):
        self.set_snr(snr_db)
    
    def set_snr(self, snr_db):
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
    
    def add_noise(self, signal):
        # Compute signal power
        signal_power = torch.mean(signal ** 2)
        
        # Compute noise power based on SNR
        noise_power = signal_power / self.snr_linear
        
        # Generate Gaussian noise
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        
        # Add noise to signal
        noisy_signal = signal + noise
        
        return noisy_signal

# Create Flask application
app = Flask(__name__)
CORS(app)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SemanticEncoder(num_classes=19, latent_dim=256).to(device)
decoder = SemanticDecoder(latent_dim=256, num_classes=19).to(device)
channel = NoisyChannel(snr_db=20)

# Set models to evaluation mode
encoder.eval()
decoder.eval()

# Load models weights (placeholder - real implementation would load actual weights)
# encoder.load_state_dict(torch.load('encoder_weights.pth'))
# decoder.load_state_dict(torch.load('decoder_weights.pth'))

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    return original_image, input_tensor.to(device)

# Calculate metrics
def calculate_metrics(original, reconstructed, latent_size):
    # Calculate Bandwidth Compression Ratio (BCR)
    # Original image: 256x256x3 = 196,608 bytes
    # Latent vector: 256 floats = 1,024 bytes (assuming 4 bytes per float)
    bcr = 1024 / 196608
    
    # Calculate PSNR
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # For a real implementation, LPIPS would require a pre-trained VGG network
    # Here we'll just provide a simulated value based on SNR
    lpips = 0.3 - (channel.snr_db / 100)  # Simplified approximation
    lpips = max(0.1, min(0.5, lpips))  # Clamp between 0.1 and 0.5
    
    return {
        'bcr': round(bcr, 5),
        'psnr': round(psnr, 2),
        'lpips': round(lpips, 2)
    }

# Convert tensor to displayable image
def tensor_to_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = tensor * 0.5 + 0.5
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy array
    array = tensor.detach().cpu().numpy()
    
    # Transpose from (C, H, W) to (H, W, C)
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    
    # Scale to [0, 255] and convert to uint8
    array = (array * 255).astype(np.uint8)
    
    return array

# Visualize semantic segmentation maps
def visualize_segmentation(seg_map_tensor):
    if seg_map_tensor.dim() == 4:
        seg_map_tensor = seg_map_tensor.squeeze(0)
    
    # Get class predictions
    _, predicted = torch.max(seg_map_tensor, dim=0)
    predicted = predicted.detach().cpu().numpy()
    
    # Define colors for visualization (19 classes in CelebAMask-HQ)
    colors = np.random.randint(0, 255, size=(19, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background
    
    # Create visualization
    vis_image = colors[predicted]
    
    return vis_image

# Convert image to base64
def img_to_base64(img):
    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img)
    
    # Ensure image is RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Encode to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# Process image through ESemCom system
def process_image(image_path):
    # Preprocess image
    original_image, input_tensor = preprocess_image(image_path)
    
    # Step 1: Semantic Encoding
    with torch.no_grad():
        semantic_maps, latent_vector = encoder(input_tensor)
    
    # Step 2: Create visualizations for the frontend
    # Original and encoded semantic visualization
    semantic_vis = visualize_segmentation(semantic_maps)
    
    # Step 3: Simulate transmitter cache (visual representation)
    cached_image = semantic_vis.copy()
    grid_size = 32
    for i in range(0, cached_image.shape[0], grid_size):
        cached_image[i:i+2, :] = [0, 255, 0]  # Green lines
    for j in range(0, cached_image.shape[1], grid_size):
        cached_image[:, j:j+2] = [0, 255, 0]
    
    # Step 4: Power normalization (normalize latent vector to constant power)
    latent_power = torch.sqrt(torch.mean(latent_vector ** 2))
    normalized_latent = latent_vector / latent_power
    
    # Visualize power normalization effect (simplified)
    pn_image = tensor_to_image(input_tensor) * 0.9  # Visual approximation
    
    # Step 5: Pass through noisy channel
    noisy_latent = channel.add_noise(normalized_latent)
    
    # Simulate the effect of noise (visual representation)
    noise_level = 10.0 / (10**(channel.snr_db/10))
    noise = np.random.normal(0, noise_level * 20, semantic_vis.shape).astype(np.int32)
    noisy_vis = np.clip(semantic_vis.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    
    # Step 6: Simulate receiver cache (visual representation)
    receiver_cached = noisy_vis.copy()
    for i in range(0, receiver_cached.shape[0], grid_size):
        receiver_cached[i:i+2, :] = [255, 0, 0]  # Red lines
    for j in range(0, receiver_cached.shape[1], grid_size):
        receiver_cached[:, j:j+2] = [255, 0, 0]
    
    # Step 7: Semantic Decoding
    with torch.no_grad():
        rec_semantic_maps, reconstructed_image = decoder(noisy_latent)
    
    # Convert reconstructed image to displayable format
    reconstructed_np = tensor_to_image(reconstructed_image)
    
    # Calculate performance metrics
    metrics = calculate_metrics(original_image, reconstructed_np, latent_vector.numel())
    
    # Return results
    result = {
        'original': img_to_base64(original_image),
        'encoded': img_to_base64(semantic_vis),
        'transmitter_cached': img_to_base64(cached_image),
        'power_normalized': img_to_base64(pn_image),
        'noisy_channel': img_to_base64(noisy_vis),
        'receiver_cached': img_to_base64(receiver_cached),
        'reconstructed': img_to_base64(reconstructed_np),
        'metrics': metrics
    }
    
    return result

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image
        try:
            result = process_image(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/set_snr', methods=['POST'])
def set_snr():
    data = request.json
    snr = data.get('snr', 20)
    channel.set_snr(snr)
    return jsonify({'status': 'success', 'snr': snr})

@app.route('/')
def index():
    return send_from_directory('dist', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('dist', path)

if __name__ == '__main__':
    app.run(debug=True)