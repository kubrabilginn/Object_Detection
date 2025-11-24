import os
import cv2
import numpy as np
from scipy.io import loadmat
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
import random

# -----------------
# 1. Sabit Tanımları ve Yollar
# -----------------
# Dosya yapınız: staj_projesi/object_detection_projem.py olduğu varsayılmıştır.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'stanford_cars') 

# Sizin klasör yapınıza göre MAT dosyası yolu
MAT_FILE = os.path.join(DATA_PATH, 'car_devkit', 'devkit', 'cars_train_annos.mat')

# Görüntü Klasörleri (Sizin yapıya göre ayarlandı)
TRAIN_IMAGES_PATH = os.path.join(DATA_PATH, 'cars_train')
NESTED_TEST_PATH = os.path.join(DATA_PATH, 'cars_test', 'cars_test') 
TEST_IMAGES_PATH = NESTED_TEST_PATH

# HOG Ayarları
HOG_ORIENTATION = 9
HOG_PIX_PER_CELL = (8, 8)
HOG_CELL_PER_BLOCK = (2, 2)
RESIZE_W = 128
RESIZE_H = 64

# KRİTİK HATA KONTROLÜ: MAT dosyası var mı?
if not os.path.exists(MAT_FILE):
    print("\n--- KRİTİK HATA: MAT ETİKET DOSYASI BULUNAMADI! ---")
    print(f"Beklenen Yol: {MAT_FILE}")
    sys.exit(1)

# -----------------
# 2. Etiketleri Okuma
# -----------------
def load_annotations(mat_file):
    """ .mat dosyasından etiketleri ve dosya yollarını yükler. """
    try:
        # cars_train_annos.mat dosyasının yapısı için doğru anahtarı kullanıyoruz
        annotations = loadmat(mat_file)['annotations'][0]
    except Exception as e:
        print(f"HATA: Etiket dosyası yüklenemedi. SciPy kurulumunu/dosya yapısını kontrol edin. Hata: {e}")
        sys.exit(1)
        
    data = []
    
    for ann in annotations:
        # Bounding Box Koordinatları
        x1 = ann['bbox_x1'][0, 0]
        y1 = ann['bbox_y1'][0, 0]
        x2 = ann['bbox_x2'][0, 0]
        y2 = ann['bbox_y2'][0, 0]
        
        # Dosya Adı (cars_annos.mat'te 'relative_im_path' idi, burada 'fname' kullanıyoruz)
        image_path = ann['fname'][0] 
        class_id = ann['class'][0, 0]
        
        data.append({
            'path': image_path,
            'bbox': (x1, y1, x2, y2),
            'class_id': class_id
        })
    return data

# -----------------
# 3. HOG Özniteliklerini Çıkarma
# -----------------
def get_hog_features(image):
    """ Görüntüden HOG öznitelik vektörünü çıkarır. """
    resized_img = cv2.resize(image, (RESIZE_W, RESIZE_H)) 
    
    features = hog(
        resized_img, 
        orientations=HOG_ORIENTATION, 
        pixels_per_cell=HOG_PIX_PER_CELL,
        cells_per_block=HOG_CELL_PER_BLOCK,
        transform_sqrt=True, 
        feature_vector=True 
    )
    return features

# ------------------------------------------------
# 4. Veri Toplama, Negatif Örnekleme ve Eğitim
# ------------------------------------------------

features = []
labels = []
annotations = load_annotations(MAT_FILE)

# Pozitif Örnekleri İşleme
print(f"Toplam {len(annotations)} pozitif örnekten uygun olanlar işleniyor...")
found_count = 0 

for ann in annotations:
    current_path = ann['path']
    base_name = os.path.basename(current_path)

    # 1. train klasöründe ara (data/stanford_cars/cars_train/00001.jpg)
    full_path = os.path.join(TRAIN_IMAGES_PATH, base_name)
    
    # 2. train'de yoksa, iç içe geçmiş test klasöründe ara (data/stanford_cars/cars_test/cars_test/00001.jpg)
    if not os.path.exists(full_path):
        full_path = os.path.join(TEST_IMAGES_PATH, base_name)
        
    if not os.path.exists(full_path):
         # Sadece ilk birkaç hata için uyarı ver
         if found_count < 10: 
             print(f"UYARI: Görüntü dosyası bulunamadı: {base_name}")
         continue 
         
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE) 

    if image is None: continue

    # Bounding Box ile arabayı kesme (crop)
    x1, y1, x2, y2 = ann['bbox']
    car_patch = image[y1:y2, x1:x2]
    
    if car_patch.size > 0 and (x2 - x1 > 0) and (y2 - y1 > 0):
        try:
            hog_features = get_hog_features(car_patch)
            features.append(hog_features)
            labels.append(1) # Pozitif (Araba) etiket
            found_count += 1
        except ValueError:
            continue
    
    if found_count % 1000 == 0 and found_count > 0:
        print(f"-> {found_count} pozitif örnek işlendi.")


# Negatif Örnekleri İşleme
print("\nNegatif örnekler oluşturuluyor...")
if found_count == 0:
    print("Pozitif örnek bulunamadığı için negatif örnek oluşturulamıyor.")
    sys.exit(1)
    
NEGATIVE_SAMPLES_COUNT = found_count // 3 
neg_count = 0

# Negatif örnekler için kullanılabilecek tüm görüntülerin listesi
all_images = os.listdir(TRAIN_IMAGES_PATH) + os.listdir(TEST_IMAGES_PATH)
random.shuffle(all_images)

while neg_count < NEGATIVE_SAMPLES_COUNT:
    base_name = random.choice(all_images)
    
    full_path = os.path.join(TRAIN_IMAGES_PATH, base_name)
    if not os.path.exists(full_path):
        full_path = os.path.join(TEST_IMAGES_PATH, base_name)
        
    if not os.path.exists(full_path):
        continue

    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    
    # ... (Negatif Örnekler oluşturma döngüsünün içinde)

    # image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    
    # ... (Negatif Örnekler oluşturma döngüsünün içinde)

    # image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None: continue

    H, W = image.shape
    patch_w, patch_h = RESIZE_W, RESIZE_H 
    
    # KRİTİK KONTROLÜ GÜÇLENDİRİYORUZ
    # Eğer görüntü genişliği veya yüksekliği yama boyutuna eşitse veya küçükse atla.
    # np.random.randint(0, x) kullanabilmek için x > 0 olmalı.
    if H <= patch_h or W <= patch_w: 
        continue 
        
    # Bu noktada W - patch_w ve H - patch_h değerleri kesinlikle 1 veya daha büyük olacaktır.
    rand_x = np.random.randint(0, W - patch_w)
    rand_y = np.random.randint(0, H - patch_h)
    
# ... (Kodun geri kalanı devam eder)
    
# ... (Kodun geri kalanı rand_y'den sonra devam eder)
    
    random_patch = image[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w]
    
    hog_features = get_hog_features(random_patch)
    features.append(hog_features)
    labels.append(0) 
    neg_count += 1
    
    if neg_count % 500 == 0 and neg_count > 0:
        print(f"-> {neg_count} negatif örnek işlendi.")

if not features:
    print("\n--- SONUÇ HATASI: Hiçbir özellik (feature) toplanamadı. ---")
    sys.exit(1)

# Veri setini ayırma
X = np.array(features)
y = np.array(labels)
print(f"\nToplam {X.shape[0]} örnek hazır (Pozitif: {labels.count(1)}, Negatif: {labels.count(0)})")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------
# 5. SVM Modelini Eğitme
# -----------------
print("SVM Modeli Eğitiliyor...")
model = LinearSVC(random_state=42, tol=1e-5, max_iter=10000) 
training_start_time = cv2.getTickCount()

try:
    model.fit(X_train, y_train)
except ValueError as e:
    print(f"EĞİTİM HATASI: {e}")
    sys.exit(1)
    
training_end_time = cv2.getTickCount()
training_time = (training_end_time - training_start_time) / cv2.getTickFrequency()
training_time_ms = training_time * 1000

print(f"\nEğitim Süresi: {training_time:.2f} saniye ({training_time_ms:.2f} ms)") 

# -----------------
# 6. Modelin Test Edilmesi
# -----------------
y_pred = model.predict(X_test)
print("\n--- HOG + SVM Performans Raporu ---")
print(classification_report(y_test, y_pred, target_names=['Negatif (Araba Değil)', 'Pozitif (Araba)']))

print("\nHOG+SVM aşaması tamamlandı. Sonuçlar yukarıdadır.")