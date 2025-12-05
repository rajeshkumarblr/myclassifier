import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from train_model import SimpleCardClassifer

# --- Configuration ---
MODEL_PATH = 'card_model.pth'
IMAGE_PATH = 'data/test/eight of diamonds/4.jpg' # Example image
DATA_DIR = './data'

# --- Load Classes ---
# We need to know the class names. Since we used ImageFolder, they are the folder names sorted alphabetically.
train_folder = os.path.join(DATA_DIR, 'train')
classes = sorted(os.listdir(train_folder))
print(f"Classes: {classes[:5]} ...")

# --- Load Model ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleCardClassifer(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set to evaluation mode
print("Model loaded successfully.")

# --- Preprocess Image ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

image = Image.open(IMAGE_PATH)
# Display image
plt.imshow(image)
plt.axis('off')
plt.show()

input_tensor = transform(image).unsqueeze(0) # Add batch dimension (1, 3, 128, 128)
input_tensor = input_tensor.to(device)

# --- Predict ---
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = classes[predicted_idx]
    confidence = probabilities[predicted_idx].item()

print(f"\nPrediction: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
