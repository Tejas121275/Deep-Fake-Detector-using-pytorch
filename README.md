# Deep-Fake-Detector-using-pytorch
Generally detect ai generate images

deepfake-detection/
│
├── app.py                
├── requirements.txt        
├── README.md
│
├── data/                   # Dataset directories (create these manually)
│   ├── real/               # Original real images
│   ├── fake/               # Original fake images
│
├── models/                 # Trained model weights
│   ├── saved_model.pth     # Main trained model
│
├── static/     # Flask static files
│   ├── Styles.css
│   └── uploads/ # User-uploaded images for prediction
│
├── templates/              # Flask HTML templates
│   └── index.html          # Main web interface template
│
├──dataset.py          # Dataset and data loader implementation
├──model.py            # Model architecture definition
├──train.py            # Training script
├──preprocess.py       # Image preprocessing script
├──diagnostic.py       # System diagnostic script
