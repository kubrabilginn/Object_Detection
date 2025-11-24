import os
import io
import time
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template # <-- render_template eklendi
from PIL import Image
import numpy as np
import sys
# ----------------------------------
# 1. Sabit Tanımlar ve Model Yükleme
# ----------------------------------
IMAGE_SIZE = 128
MODEL_PATH = 'simple_cnn_car_detector.pth'
CLASSES = ['Negatif (Arka Plan)', 'Pozitif (Araba)']
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
    sys.exit(1)

# Basit CNN Mimarisi (cnn_object_classification.py dosyasından kopyalanmıştır)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128), 
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

# Modeli belleğe yükle
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() # Çıkarım moduna al
print(f"\nModel başarıyla yüklendi: {DEVICE} cihazında.")


# ----------------------------------
# 2. Flask Uygulaması
# ----------------------------------
app = Flask(__name__)

def preprocess_image(image_data):
    """ Web'den gelen görüntüyü CNN girişi için hazırlar. """
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Görüntüyü tensöre çevir ve normalize et
    image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) # (C, H, W)
    
    # Normalizasyon değerleri
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image_tensor = (image_tensor / 255.0 - mean) / std
    
    # Modelin beklediği batch boyutunu ekle (1, C, H, W)
    return image_tensor.unsqueeze(0).to(DEVICE)

@app.route('/')
def index():
    # templates/index.html dosyasını döndürür
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    file = request.files['file']
    img_bytes = file.read()
    
    try:
        # Görüntüyü işle
        input_tensor = preprocess_image(img_bytes)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # En yüksek olasılığa sahip sınıfı bul
            confidence, predicted_class_index = torch.max(probabilities, 0)
            
            # Sonuçları al
            predicted_class = CLASSES[predicted_class_index.item()]
            confidence_score = confidence.item()

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000 # Gecikme süresi (milisaniye)

    # Sonuçları JSON formatında döndür
    return jsonify({
        'prediction': predicted_class,
        'confidence': f'{confidence_score:.4f}',
        'latency_ms': f'{latency_ms:.2f}'
    })

if __name__ == '__main__':
    # Flask sunucusunu başlat
    # Thread sayısı 1 tutulmuştur, MPS kaynaklarını çakıştırmamak için
    app.run(host='0.0.0.0', port=5002)