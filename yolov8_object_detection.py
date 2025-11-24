import os
import torch
import time
from ultralytics import YOLO
import sys 

# -----------------
# 1. Sabit Tanımları ve Yollar
# -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
YAML_PATH = os.path.join(PROJECT_ROOT, 'stanford_cars.yaml')

# YOLO modelinin eğitileceği cihaz
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" 

# Model ayarları (Hızlandırılmış Test için ayarlandı)
MODEL_SIZE = 'yolov8n.pt' # En küçük model (nano)
EPOCHS = 15               # Hızlandırılmış Test için 15 epoch
IMG_SIZE = 640            
BATCH_SIZE = 16           

print(f"Cihaz: {DEVICE}")
print(f"Model: {MODEL_SIZE}, Epoch: {EPOCHS}, Batch Size: {BATCH_SIZE}")

# -----------------
# 2. YOLO Modelini Yükleme ve Eğitim
# -----------------
try:
    # Önceden eğitilmiş nano modelini yükle
    model = YOLO(MODEL_SIZE)  
    
    start_time = time.time()
    
    print("\n--- YOLOv8 Eğitimi Başlatılıyor (15 Epoch) ---")
    
    # Modeli eğit
    results = model.train(
        data=YAML_PATH,  
        epochs=EPOCHS, 
        imgsz=IMG_SIZE, 
        batch=BATCH_SIZE, 
        device=DEVICE,
        name='yolov8_car_detector', 
        save=True,               
        workers=1,               # M2'de uyumluluk için
        optimizer='AdamW',
        cos_lr=True              
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n--- YOLOv8 Eğitim Tamamlandı ---")
    print(f"Eğitim Süresi: {training_time:.2f} saniye")
    
except Exception as e:
    print(f"YOLOv8 eğitiminde kritik hata: {e}")
    sys.exit(1)

# -----------------
# 3. Test/Doğrulama Metriklerini Alma
# -----------------
try:
    print("\n--- Model Performansı Doğrulanıyor (Val Seti) ---")
    
    # Eğitim çıktıları genellikle 'runs/detect/yolov8_car_detector/weights/best.pt' altında kaydedilir
    # Sonuçların kaydedildiği klasörü bul (Eğitim sonunda otomatik oluşturulur)
    
    # Ultralytics, eğitim sonunda metrikleri zaten döndürdüğü için, 
    # ek bir 'val' komutuna gerek kalmadan sonuçları çekelim.
    
    metrics = results.box
    
    print("\n--- YOLOv8 Performans Metrikleri ---")
    print(f"mAP50 (Ortalama Hassasiyet @ 0.5 IoU): {metrics.map50:.4f}") 
    print(f"mAP50-95 (Genel Ortalama Hassasiyet): {metrics.map:.4f}")
    print(f"Precision (P): {metrics.mp:.4f}")
    print(f"Recall (R): {metrics.mr:.4f}")
    
except Exception as e:
    print(f"YOLOv8 doğrulamasında hata (Eğitim sonrası): {e}")