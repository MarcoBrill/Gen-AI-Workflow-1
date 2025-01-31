import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from torchvision import transforms
from dain import DAIN

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "stabilityai/stable-diffusion-2-1"
IMAGE_SIZE = (512, 512)
VIDEO_FRAME_RATE = 30

# Load Stable Diffusion models
def load_models():
    print("Loading Stable Diffusion models...")
    text_to_image = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    image_to_image = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    return text_to_image, image_to_image

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Generate professional image
def generate_professional_image(image_path, prompt="professional high-quality image"):
    text_to_image, image_to_image = load_models()
    input_image = preprocess_image(image_path)
    
    print("Generating professional image...")
    with torch.autocast(DEVICE):
        output_image = image_to_image(
            prompt=prompt,
            init_image=input_image,
            strength=0.75,
            guidance_scale=7.5,
        ).images[0]
    
    output_path = "output/professional_image.png"
    output_image.save(output_path)
    print(f"Professional image saved to {output_path}")
    return output_path

# Enhance video frames using DAIN
def enhance_video(video_path):
    print("Enhancing video frames...")
    dain_model = DAIN().to(DEVICE)
    output_video_path = "output/enhanced_video.mp4"
    
    # Process video frames
    dain_model.interpolate_video(video_path, output_video_path, frame_rate=VIDEO_FRAME_RATE)
    print(f"Enhanced video saved to {output_video_path}")
    return output_video_path

# Main function
def main():
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Example usage
    image_path = "input/example_image.jpg"
    video_path = "input/example_video.mp4"
    
    # Generate professional image
    professional_image_path = generate_professional_image(image_path)
    
    # Enhance video
    enhanced_video_path = enhance_video(video_path)

if __name__ == "__main__":
    main()
