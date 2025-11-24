# 🚘 Derin Öğrenme ve Geleneksel Yöntemlerle Araç Nesne Tanıma 

Bu proje, **Stanford Cars Dataset** üzerinde, hem geleneksel makine öğrenimi tekniklerini hem de modern derin öğrenme yaklaşımlarını kullanarak araç nesne tanıma (Object Detection) performansını karşılaştırmayı amaçlamaktadır.

Proje, özellikle **PyTorch/Ultralytics üzerinde gerçekleştirilmiş olup, farklı algoritmaların eğitim süreleri ve doğruluk (Accuracy/mAP) metrikleri detaylı olarak analiz edilmiştir.

## 📊 Karşılaştırılan Algoritmalar

Bu karşılaştırma tablosu, her algoritmanın eğitim süresini ve ulaştığı nihai performans metriklerini özetlemektedir.

| Algoritma | Görev | Eğitim Süresi | Metrik | Metrik Değeri |
| :--- | :--- | :--- | :--- | :--- |
| **HOG + KNN** | Sınıflandırma | 0.02 saniye | Accuracy | %66.0 |
| **HOG + SVM** | Sınıflandırma | 16.41 saniye | Accuracy | %71.0 |
| **Basit CNN** | Sınıflandırma | **314.04 saniye** | Accuracy | **%85.0** |
| **YOLOv8n (Nesne Tanıma)** | Nesne Tanıma & Konumlandırma | **1.855 saat** | mAP50 | **%9.08** |

1.  **En Yüksek Başarı:** Sadece **sınıflandırma** görevi için en yüksek başarıyı (%85.0 Accuracy) Basit CNN modeli elde etmiştir.
2.  **YOLOv8 Düşük mAP Değeri Analizi:** YOLOv8'in mAP50 değerinin (%9.08) bu kadar düşük olması beklenmediktir. Bunun temel nedenleri, kaynak kısıtlamaları nedeniyle eğitim süresinin **15 epoch** ile sınırlı kalması ve mAP'nin sadece doğru sınıfı değil, aynı zamanda doğru **konumu (Bounding Box)** da gerektiren çok daha zorlu bir metrik olmasıdır.
3.  **Hız Farkı:** Geleneksel algoritmalar saniyeler içinde eğitilirken (Örn: HOG+SVM 16.41 saniye), Derin Öğrenme modelleri (CNN 314.04 saniye, YOLOv8 1.855 saat) çok daha fazla eğitim süresi gerektirmiştir.


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
