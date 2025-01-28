import torch
import numpy as np
from PIL import Image
from torch.utils.vae import MeanShift

# Create empty latent (8x smaller than target)
latent_size = 64  # For 512x512 images
latents = np.zeros((latent_size, latent_size), dtype=np.float32)

def VAE():
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2)
    )
    
    decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        MeanShift(0.5)  # Add noise during decoding
    )
    
    return torch.nn.Sequential(encoder, decoder)

# Initialize VAE model and move to device (replace 'cuda' with 'cpu' if using CPU)
device = 'cuda'  # Replace with 'cpu' for CPU
vaemodel = VAE().to(device)

# Encode the empty latent to get z (this will be all zeros)
z = vaemodel(torch.unsqueeze(latents, 0).to(device))  # Input tensor shape: batch_size=1

# Decode the z to get the final image with noise (don't remove noise for visualization purposes)
decoded_image_with_noise = vaemodel.decode(z.squeeze(0))

# Convert tensor to PIL Image and save
image = ToTensor()(decoded_image_with_noise).squeeze().cpu()
Image.fromarray(np.array(image, dtype=np.uint8)).save('empty_latent.png')

