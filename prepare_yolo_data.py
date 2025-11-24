import os
import cv2
import numpy as np
from scipy.io import loadmat
import sys
import random

# -----------------
# 1. Sabit Tanımları ve Yollar
# -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'stanford_cars') 
MAT_FILE = os.path.join(DATA_PATH, 'car_devkit', 'devkit', 'cars_train_annos.mat')
TRAIN_IMAGES_PATH = os.path.join(DATA_PATH, 'cars_train')
NESTED_TEST_PATH = os.path.join(DATA_PATH, 'cars_test', 'cars_test') 

# YOLO verilerinin kaydedileceği klasörler
YOLO_ROOT = os.path.join(DATA_PATH, 'yolo_data')
YOLO_IMAGES = os.path.join(YOLO_ROOT, 'images')
YOLO_LABELS = os.path.join(YOLO_ROOT, 'labels')
YOLO_TRAIN_IMAGES = os.path.join(YOLO_IMAGES, 'train')
YOLO_VAL_IMAGES = os.path.join(YOLO_IMAGES, 'val')
YOLO_TRAIN_LABELS = os.path.join(YOLO_LABELS, 'train')
YOLO_VAL_LABELS = os.path.join(YOLO_LABELS, 'val')

# Tek bir sınıfımız var: Araba (Car)
CLASS_ID = 0 

# Klasörleri oluştur
os.makedirs(YOLO_TRAIN_IMAGES, exist_ok=True)
os.makedirs(YOLO_VAL_IMAGES, exist_ok=True)
os.makedirs(YOLO_TRAIN_LABELS, exist_ok=True)
os.makedirs(YOLO_VAL_LABELS, exist_ok=True)


# -----------------
# 2. Koordinat Dönüşüm Fonksiyonu
# -----------------
def convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max):
    """ PASCAL VOC/MATLAB koordinatlarını YOLO formatına çevirir (normalize edilmiş). """
    
    # 1. Ortanca nokta (center x, center y)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    
    # 2. Genişlik ve Yükseklik
    width = x_max - x_min
    height = y_max - y_min
    
    # 3. Normalizasyon (0 ile 1 arasına)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

# -----------------
# 3. Ana Dönüşüm Mantığı
# -----------------
def process_annotations():
    print("Etiket dosyaları yükleniyor...")
    try:
        annotations = loadmat(MAT_FILE)['annotations'][0]
    except Exception as e:
        print(f"HATA: Etiket dosyası yüklenemedi: {e}")
        sys.exit(1)
        
    print(f"Toplam {len(annotations)} etiket işlenecek.")
    
    # Veriyi eğitim ve doğrulama (validation) setlerine ayır (Genellikle %80/%20)
    train_split_count = int(len(annotations) * 0.8)
    random.shuffle(annotations)
    train_annotations = annotations[:train_split_count]
    val_annotations = annotations[train_split_count:]

    all_annotations = {'train': train_annotations, 'val': val_annotations}
    
    for split_name, ann_list in all_annotations.items():
        print(f"\n{split_name.upper()} seti için {len(ann_list)} etiket işleniyor...")
        
        current_img_dir = YOLO_TRAIN_IMAGES if split_name == 'train' else YOLO_VAL_IMAGES
        current_label_dir = YOLO_TRAIN_LABELS if split_name == 'train' else YOLO_VAL_LABELS

        for idx, ann in enumerate(ann_list):
            
            # Etiket bilgilerini çekme
            x1 = ann['bbox_x1'][0, 0]; y1 = ann['bbox_y1'][0, 0]
            x2 = ann['bbox_x2'][0, 0]; y2 = ann['bbox_y2'][0, 0]
            base_name = os.path.basename(ann['fname'][0])
            img_filename, _ = os.path.splitext(base_name)
            
            # Görüntü Yolu Bulma (Önceki adımlarda doğrulanan mantık)
            full_path = os.path.join(TRAIN_IMAGES_PATH, base_name)
            if not os.path.exists(full_path):
                full_path = os.path.join(NESTED_TEST_PATH, base_name)
                
            if not os.path.exists(full_path):
                 continue 
                 
            # Görüntüyü okuyarak boyutlarını al
            img = cv2.imread(full_path)
            if img is None:
                continue 
                
            H, W, _ = img.shape
            
            # YOLO formatına çevir
            x_center, y_center, width, height = convert_to_yolo_format(W, H, x1, y1, x2, y2)
            
            # YOLO etiket satırını oluştur
            yolo_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            # Etiket dosyasını kaydet
            label_filepath = os.path.join(current_label_dir, f"{img_filename}.txt")
            with open(label_filepath, 'w') as f:
                f.write(yolo_line)
                
            # Görüntüyü YOLO klasörüne kopyala
            cv2.imwrite(os.path.join(current_img_dir, base_name), img)
            
            if (idx + 1) % 1000 == 0:
                print(f"-> {idx + 1} etiket ve görüntü kopyalandı.")

    print("\nYOLO formatı hazırlığı tamamlandı. Veri Yolu:", YOLO_ROOT)


if __name__ == '__main__':
    process_annotations()