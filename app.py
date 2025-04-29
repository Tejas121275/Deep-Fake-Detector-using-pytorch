from flask import Flask, request, render_template
import torch
from model import DeepfakeDetector
from PIL import Image
import torchvision.transforms as transforms
import os
from facenet_pytorch import MTCNN
import numpy as np

app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load('models/saved_model.pth', map_location=device))
model.eval()

# Force MTCNN to run on CPU to avoid torchvision::nms CUDA error
mtcnn = MTCNN(image_size=224, margin=20, device='cpu')

# Standard transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test-time augmentation transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    image_url = None
    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file uploaded'
        else:
            file = request.files['file']
            if file.filename == '':
                message = 'No file selected'
            elif file:
                try:
                    # Save uploaded image with a unique filename to avoid conflicts
                    os.makedirs('static/uploads', exist_ok=True)
                    filename = os.path.join('static/uploads', file.filename)
                    file.save(filename)
                    image = Image.open(filename).convert('RGB')

                    # Face detection with MTCNN on CPU
                    img_array = np.array(image)
                    boxes, _ = mtcnn.detect(img_array)
                    if boxes is not None and len(boxes) > 0:
                        x1, y1, x2, y2 = boxes[0].astype(int)
                        image = image.crop((x1, y1, x2, y2))
                    else:
                        message = 'No face detected'
                        image_url = os.path.relpath(filename, 'static').replace('\\', '/')
                        return render_template('index.html', message=message, image_url=image_url)

                    # Test-time augmentation
                    images = [transform(image)]
                    for tta_transform in tta_transforms:
                        images.append(tta_transform(image))
                    images = torch.stack(images).to(device)

                    # Prediction with TTA
                    with torch.no_grad():
                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        avg_prob = probs.mean().item()
                        result = 'Fake' if avg_prob > 0.5 else 'Real'
                        confidence = avg_prob if result == 'Fake' else 1 - avg_prob

                    message = f'Prediction: {result} (Confidence: {confidence:.2%})'
                    image_url = os.path.relpath(filename, 'static').replace('\\', '/')
                except Exception as e:
                    message = f'Error processing image: {str(e)}'
                    image_url = os.path.relpath(filename, 'static').replace('\\', '/') if 'filename' in locals() else None
        return render_template('index.html', message=message, image_url=image_url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)