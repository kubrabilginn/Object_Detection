# 🚘 Derin Öğrenme ve Geleneksel ML Yöntemleriyle Araç Nesne Tanıma Karşılaştırması ve Flask API entegrasyonu

Bu proje, **Stanford Cars Dataset** üzerinde hem geleneksel hem de modern derin öğrenme yaklaşımlarını kullanarak **Araç Nesne Tanıma** performansını karşılaştırmayı amaçlamaktadır. Proje, özellikle **Macbook M2 Air'in MPS (Metal Performance Shaders) hızlandırması** kullanılarak PyTorch ile optimize edilmiştir.

Modelin pratik uygulanabilirliğini göstermek amacıyla, **Basit CNN** modeli **Flask** web sunucusu üzerinden bir API'ye dönüştürülmüştür.

## 📊 Sonuçlar ve Performans Analizi

Test edilen algoritmaların eğitim süreleri ve elde edilen doğruluk (Accuracy/mAP) metrikleri aşağıdadır.

| Algoritma | Görev Tipi | Eğitim Süresi | Metrik | Metrik Değeri |
| :--- | :--- | :--- | :--- | :--- |
| **HOG + KNN** | Sınıflandırma | 0.02 saniye | Accuracy | %66.0 |
| **HOG + SVM** | Sınıflandırma | 16.41 saniye | Accuracy | %71.0 |
| **Basit CNN** | Sınıflandırma | 314.04 saniye | **Accuracy** | **%85.0** |
| **YOLOv8n** | Nesne Tanıma (Konumlandırma) | **1.855 saat** | mAP50 | %9.08 |

### Sonuçların Yorumlanması

1.  **En Yüksek Başarı (Sınıflandırma):** **Basit CNN** modeli, otomatik öznitelik çıkarımı sayesinde en yüksek doğruluk oranına (%85.0) ulaşmıştır.
2.  **YOLOv8 Analizi:** YOLOv8'in mAP50 değerinin düşük çıkması (%9.08), **konumlandırma** gereksiniminden ve **kısa epoch** (15) sayısından kaynaklanmıştır. M2'de eğitim süresi $1.85$ saat olarak kaydedilmiştir.
3.  **Geleneksel Yöntemler:** HOG öznitelikleri, SVM ile sınıflandırıldığında ($71\%$), hızlı bir temel başarı (baseline) sağlamıştır. 

### Derin Öğrenme Teorisi
* **Transfer Learning:** Proje, **fine-tuning** kavramının pratik uygulamasını ve model formatlarının (`.pt`, `.h5`) araştırılmasını kapsamıştır.

### Web Servis Entegrasyonu
* **Gereksinim:** Yüksek doğruluklu tahminin pratik uygulaması.
* **Çözüm:** **Flask** framework'ü kullanılarak bir API oluşturulmuştur.
* **Uç Nokta:** `http://127.0.0.1:5002/predict`
* **Ölçülen Performans:** Modelin web üzerinden tahmin yapma hızı (latency) ölçülerek, çıkarım (inference) performansı belgelenmiştir.

## 🚀 Kurulum ve Çalıştırma Adımları

**Not:** Bu talimatlar, [yolo\_final\_env] ortamınızı oluşturduğunuzu varsayar.

### API Sunucusunu Başlatma

Flask API'sini arka planda çalıştırmak için:

```bash
# OpenMP çakışmasını önle
export KMP_DUPLICATE_LIB_OK=TRUE

# API Sunucusunu Başlat (Port 5002)
python app.py
