# 🚘 Derin Öğrenme ve Geleneksel Yöntemlerle Araç Nesne Tanıma 

Bu proje, **Stanford Cars Dataset** üzerinde, hem geleneksel makine öğrenimi tekniklerini hem de modern derin öğrenme yaklaşımlarını kullanarak araç nesne tanıma (Object Detection) performansını karşılaştırmayı amaçlamaktadır.

Proje, özellikle **PyTorch/Ultralytics (Macbook M2 Air MPS hızlandırması ile)** üzerinde gerçekleştirilmiş olup, farklı algoritmaların eğitim süreleri ve doğruluk (Accuracy/mAP) metrikleri detaylı olarak analiz edilmiştir.

## 📊 Karşılaştırılan Algoritmalar

Projede, aynı veri seti ve görev için üç ana kategoride model karşılaştırması yapılmıştır.

| Kategori | Algoritma | Görev | Temel Öznitelik | Genel Doğruluk (Accuracy/mAP) |
| :--- | :--- | :--- | :--- | :--- |
| **Geleneksel ML (Feature Engineering)** | HOG + SVM | Sınıflandırma | El ile kodlanmış (HOG) | %71.0 |
| **Geleneksel ML (Feature Engineering)** | HOG + KNN | Sınıflandırma | El ile kodlanmış (HOG) | %66.0 |
| **Derin Öğrenme (Sınıflandırma)** | Basit CNN (PyTorch) | Sınıflandırma | Otomatik (Evrişimli Katmanlar) | **%85.0 (Accuracy)** |
| **Derin Öğrenme (Nesne Tanıma)** | YOLOv8n (Ultralytics) | Nesne Tanıma & Konumlandırma | Otomatik (Tek Aşamalı Algılayıcı) | **%87.0 (mAP50)** |


## ✨ Proje Aşamaları

1.  **Veri Seti Hazırlığı:** Stanford Cars Dataset'in etiketlerinin (.mat) ayrıştırılması ve görüntülerden pozitif/negatif örneklerin çıkarılması.
2.  **Geleneksel Algoritmalar:** HOG öznitelik vektörlerinin oluşturulması ve SVM/KNN modellerinin eğitimi.
3.  **Basit CNN Eğitimi:** PyTorch framework'ü ile basit bir CNN mimarisinin oluşturulması ve ikili sınıflandırma (Araba vs. Arka Plan) için eğitimi. **(Eğitim Süresi: 314.04 saniye)**
4.  **YOLOv8 Hazırlığı:** Bounding Box koordinatlarının YOLO formatına (.txt) dönüştürülmesi.
5.  **YOLOv8 Eğitimi:** YOLOv8n modelinin MPS (Metal Performance Shaders) hızlandırması kullanılarak eğitilmesi ve mAP (Mean Average Precision) metriklerinin analizi.

## Kurulum ve Çalıştırma

Bu projeyi yerel olarak çalıştırmak için aşağıdaki adımları izleyin:

### Gereksinimler

* Python 3.10+
* Miniconda / Anaconda

### Ortam Kurulumu

```bash
# 1. Yeni Conda ortamı oluşturma
conda create -n yolo_project_env python=3.10
conda activate yolo_project_env

# 2. Gerekli kütüphaneleri kurma
conda install scipy numpy scikit-learn opencv
pip install torch torchvision torchaudio ultralytics

# 3. Görüntüleri ve etiketleri hazırlama (Manuel indirme gereklidir)
# Görüntüleri ve etiketleri projenizin `data/stanford_cars/` klasörüne yerleştirin.
python prepare_yolo_data.py # YOLO etiketlerini hazırlar
