import os
import numpy as np
import cv2
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import time

# -----------------
# 1. Sabit Tanımları ve Yollar
# -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'stanford_cars') 
MAT_FILE = os.path.join(DATA_PATH, 'car_devkit', 'devkit', 'cars_train_annos.mat')
TRAIN_IMAGES_PATH = os.path.join(DATA_PATH, 'cars_train')
NESTED_TEST_PATH = os.path.join(DATA_PATH, 'cars_test', 'cars_test') 
TEST_IMAGES_PATH = NESTED_TEST_PATH

IMAGE_SIZE = 128 
NUM_EPOCHS = 5 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

if not os.path.exists(MAT_FILE):
    print("\n--- KRİTİK HATA: MAT ETİKET DOSYASI BULUNAMADI! ---")
    sys.exit(1)

# -----------------
# 2. Veri Yükleyici (Dataset) Tanımlama
# -----------------
def load_annotations(mat_file):
    """ .mat dosyasından etiketleri ve dosya yollarını yükler. """
    try:
        annotations = loadmat(mat_file)['annotations'][0]
    except Exception as e:
        print(f"HATA: Etiket dosyası yüklenemedi: {e}")
        sys.exit(1)
    
    data = []
    for ann in annotations:
        x1 = ann['bbox_x1'][0, 0]; y1 = ann['bbox_y1'][0, 0]
        x2 = ann['bbox_x2'][0, 0]; y2 = ann['bbox_y2'][0, 0]
        image_path = ann['fname'][0] 
        
        data.append({
            'path': image_path,
            'bbox': (x1, y1, x2, y2),
            'label': 1 
        })
    return data

class CarDataset(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        base_name = ann['path']
        
        # Görüntü Yolu Arama
        full_path = os.path.join(TRAIN_IMAGES_PATH, base_name)
        if not os.path.exists(full_path):
            full_path = os.path.join(TEST_IMAGES_PATH, base_name)
            
        if not os.path.exists(full_path):
             return self.__getitem__(np.random.randint(len(self)))

        image = cv2.imread(full_path) 
        
        # --- HATA ÖNLEME KONTROLÜ (KRİTİK) ---
        if image is None:
            # Görüntü bozuksa, yüklenemezse veya eksikse, yeni bir örnek al
            return self.__getitem__(np.random.randint(len(self)))
        # --------------------------------------

        x1, y1, x2, y2 = ann['bbox']

        # Eğer negatif örnekse (Bounding box 0 ise), rastgele bir yama kes
        if ann['label'] == 0:
            H, W, _ = image.shape
            
            if H <= IMAGE_SIZE or W <= IMAGE_SIZE: 
                return self.__getitem__(np.random.randint(len(self)))
                
            rand_x = np.random.randint(0, W - IMAGE_SIZE)
            rand_y = np.random.randint(0, H - IMAGE_SIZE)
            
            car_patch = image[rand_y:rand_y + IMAGE_SIZE, rand_x:rand_x + IMAGE_SIZE]

        # Eğer pozitif örnekse, sınırlayıcı kutuya göre kes
        else:
            car_patch = image[y1:y2, x1:x2]
            
            if car_patch is None or car_patch.size == 0 or car_patch.shape[0] < 10 or car_patch.shape[1] < 10:
                return self.__getitem__(np.random.randint(len(self)))
            
            car_patch = cv2.resize(car_patch, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Dönüşüm ve Normalizasyon
        image_rgb = cv2.cvtColor(car_patch, cv2.COLOR_BGR2RGB)
        label = ann['label']
        
        image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) # (C, H, W)
        
        # Normalizasyon (Manuel)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image_tensor = (image_tensor / 255.0 - mean) / std
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

# -----------------
# 3. Basit CNN Mimarisi Tanımlama
# -----------------
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

# ------------------------------------------------
# 4. Veri Toplama ve Ayırma
# ------------------------------------------------
annotations = load_annotations(MAT_FILE)

positive_annotations = []
for ann in annotations:
    ann['path'] = os.path.basename(ann['path']) 
    ann['label'] = 1 
    positive_annotations.append(ann)

print("Negatif örnekler oluşturuluyor...")
negative_annotations = []
NEGATIVE_SAMPLES_COUNT = len(positive_annotations)
all_images = os.listdir(TRAIN_IMAGES_PATH) + os.listdir(TEST_IMAGES_PATH)
random.shuffle(all_images)

while len(negative_annotations) < NEGATIVE_SAMPLES_COUNT:
    base_name = random.choice(all_images)
    
    negative_annotations.append({
        'path': base_name, 
        'bbox': (0, 0, 0, 0), 
        'label': 0 
    })

all_data = positive_annotations + negative_annotations
random.shuffle(all_data)

train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
print(f"Toplam Pozitif/Negatif Örnek: {len(positive_annotations)}/{len(negative_annotations)}")
print(f"Eğitim Seti: {len(train_data)}, Test Seti: {len(test_data)}")

train_dataset = CarDataset(train_data)
test_dataset = CarDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------
# 5. Eğitim Döngüsü
# -----------------
model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel {DEVICE} cihazında eğitiliyor... (Eğitim Başlatılıyor)")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

end_time = time.time()
training_time = end_time - start_time
print(f"\nEğitim Süresi: {training_time:.2f} saniye")


# -----------------
# 6. Test ve Raporlama
# -----------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- Basit CNN Performans Raporu ---")
print(classification_report(all_labels, all_preds, target_names=['Negatif (Arka Plan)', 'Pozitif (Araba)']))

print("\nCNN aşaması tamamlandı.")