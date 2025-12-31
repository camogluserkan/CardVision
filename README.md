# TC Kimlik Kartı Tanıma ve OCR Sistemi

Bu proje, **Dijital Görüntü İşleme** dersi kapsamında geliştirilmiştir. TC Kimlik kartlarının ön yüzünü işleyerek üzerindeki metinleri (Ad, Soyad, TC vb.) anlamlı veriye dönüştürür ve JSON formatında çıktı üretir.

Proje, son kullanıcılar için kullanıcı dostu bir **Grafik Arayüz (GUI)** sunarken, geliştiriciler ve detaylı analizler için kapsamlı **test araçları (CLI)** da içermektedir.

## 🚀 Özellikler

* **Görüntü İşleme:** Perspektif Düzeltme (Warp Perspective), Akıllı Yön Bulma (Auto-Rotation), Gürültü Temizleme ve Thresholding.
* **Segmentasyon:** Blok tabanlı metin ayrıştırma algoritması ile kimlik üzerindeki alanların tespiti.
* **OCR:** Tesseract motoru entegrasyonu ile yüksek doğrulukta metin okuma.
* **Veri Çözümleme (Parsing):** OCR'dan gelen ham metni kurallı bir şekilde işleyip (Ad, Soyad, TC, Doğum Tarihi vb.) JSON formatına dönüştüren parser modülü.
* **Çoklu Çalışma Modu:** GUI (Arayüz), CLI (Tekli) ve Batch (Toplu Test) desteği.

## 🛠️ Kurulum

Projenin çalışması için Python kütüphaneleri ve Tesseract OCR motoru gereklidir.

### 1. Python Kütüphaneleri
Gerekli temel paketleri (OpenCV, Numpy, Pytesseract) yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Tesseract-OCR Kurulumu
Sistemin metinleri okuyabilmesi için Tesseract OCR motorunun bilgisayarda yüklü olması şarttır.

Windows: Tesseract Installer adresinden indirip kurun. Kurulum yolunu (Path) sisteme eklemeyi unutmayın.

Linux (Ubuntu/Debian):
```bash
sudo apt-get install tesseract-ocr-tur
```
### 3. Arayüz Kütüphanesi (Sadece Linux İçin)
Windows ve macOS kullanıcılarında tkinter Python ile yüklü gelir. Linux kullanıyorsanız şu komutu çalıştırmanız gerekir:
```bash
sudo apt-get install python3-tk
```

### 💻 Kullanım Modları
Proje üç farklı senaryo için üç farklı çalıştırma dosyası sunar:

### 1. Grafik Arayüz (Son Kullanıcı Modu)
Görsel bir pencere üzerinden resim seçip sonuçları anlık görmek için kullanılır. Sunumlar için idealdir.

```bash
python gui_app.py
```

Nasıl Kullanılır: "Resim Seç" butonuna basın -> Dosyayı seçin -> "BAŞLAT" butonuna basın.

### 2. Tekli Test Modu (Geliştirici Modu)
Kod üzerinde değişiklik yapıldığında tek bir resim ile hızlı deneme yapmak için kullanılır.

```bash
python main.py
```

Not: Bu dosya, kodun içinde sabit tanımlanan (INPUT_IMAGE_PATH) resmi işler ve sonuçları terminale basar.

### 3. Toplu Test ve Hata Ayıklama Modu (Batch Processing)
Klasördeki tüm resimleri sırayla işler. Hata alan dosyaları otomatik olarak _FAIL veya _CRASH etiketli klasörlere ayırır. Sistemin genel başarısını ölçmek için idealdir.

```bash
python tests.py
```

Girdi: input/ klasöründeki tüm resimler.

Çıktı: test_outputs/ klasöründe her resim için ayrı rapor oluşturulur.

### 📂 Dosya ve Klasör Yapısı
#### gui_app.py: Projenin görsel arayüzü (Tkinter tabanlı).

#### main.py: Tekil dosya çalıştırma ve temel entegrasyon betiği.

#### tests.py: Toplu dosya işleme, klasör temizleme ve hata yönetimi betiği.

#### image_processor.py: Görüntü işleme çekirdeği (OpenCV fonksiyonları: Normalizasyon, Segmentasyon).

#### id_parser.py: OCR çıktılarını anlamlandıran mantık modülü (Regex ve kural tabanlı).

#### input/: Test edilecek ham kimlik fotoğrafları buraya konulur.

#### output_gui/: Arayüzden yapılan işlemlerin geçici çıktıları.

#### test_outputs/: Toplu test sonuçlarının raporlandığı dizin.