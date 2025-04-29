from facenet_pytorch import MTCNN
from PIL import Image
import os
import numpy as np

# Force CPU to avoid NMS CUDA error
device = 'cpu'
mtcnn = MTCNN(image_size=224, margin=20, device=device)
input_dirs = ['data/real', 'data/fake']  # Update to your local paths (e.g., C:/deepfake_data/real)
output_dirs = ['data/real_cropped', 'data/fake_cropped']

for input_dir, output_dir in zip(input_dirs, output_dirs):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                img_array = np.array(image)
                boxes, _ = mtcnn.detect(img_array)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    image = image.crop((x1, y1, x2, y2))
                    image.save(os.path.join(output_dir, img_name))
                else:
                    print(f"No face detected in {img_name}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")