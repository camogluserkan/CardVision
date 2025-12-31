import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import cv2
import pytesseract
import os
import json
import threading # Arayüz donmasın diye işlemi arkada yapmak için

# Mevcut modüllerimizi çağırıyoruz
from image_processor import normalize_id_card, segment_fields_from_blobs
from id_parser import parse_id_card_data

# --- OCR AYARLARI ---
# Windows kullanıyorsanız ve Tesseract Path hatası alırsanız burayı açın:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class IDCardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TC Kimlik Kartı Okuma Sistemi")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        # Seçilen dosya yolu
        self.file_path = None
        self.output_dir = "output_gui"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.setup_ui()

    def setup_ui(self):
        # --- BAŞLIK ---
        lbl_title = tk.Label(self.root, text="TC Kimlik OCR Sistemi", font=("Helvetica", 16, "bold"), fg="#333")
        lbl_title.pack(pady=15)

        # --- DOSYA SEÇME ALANI ---
        frame_input = tk.Frame(self.root, pady=10)
        frame_input.pack()

        self.btn_select = tk.Button(frame_input, text="Resim Seç...", command=self.select_file, width=15, bg="#e1e1e1")
        self.btn_select.pack(side=tk.LEFT, padx=10)

        self.lbl_path = tk.Label(frame_input, text="Dosya seçilmedi", fg="gray", width=50, anchor="w", bg="white", relief="sunken")
        self.lbl_path.pack(side=tk.LEFT)

        # --- İŞLEM BUTONU ---
        self.btn_process = tk.Button(self.root, text="BAŞLAT / OKU", command=self.start_processing_thread, 
                                     font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", width=20, state=tk.DISABLED)
        self.btn_process.pack(pady=15)

        # --- SONUÇ EKRANI (LOG) ---
        lbl_log = tk.Label(self.root, text="İşlem Kaydı ve Sonuçlar:", font=("Helvetica", 10, "bold"))
        lbl_log.pack(anchor="w", padx=20)

        self.txt_log = scrolledtext.ScrolledText(self.root, width=80, height=20, font=("Consolas", 10))
        self.txt_log.pack(padx=20, pady=5)

        # --- ALT BİLGİ ---
        lbl_footer = tk.Label(self.root, text="Digital Image Processing Projesi", fg="gray", font=("Arial", 8))
        lbl_footer.pack(side=tk.BOTTOM, pady=10)

    def log(self, message):
        """Arayüzdeki metin kutusuna yazı yazar."""
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END) # Otomatik aşağı kaydır

    def select_file(self):
        """Kullanıcıya dosya seçtirir."""
        filetypes = (("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp"), ("Tüm Dosyalar", "*.*"))
        path = filedialog.askopenfilename(title="Kimlik Resmi Seç", filetypes=filetypes)
        
        if path:
            self.file_path = path
            self.lbl_path.config(text=path, fg="black")
            self.btn_process.config(state=tk.NORMAL) # Butonu aktif et
            self.txt_log.delete(1.0, tk.END) # Logu temizle
            self.log(f"Resim seçildi: {os.path.basename(path)}")
            self.log("İşleme başlamak için yeşil butona basın.")

    def preprocess_for_ocr(self, img):
        """OCR öncesi görüntü iyileştirme (Main.py'den alındı)"""
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        padded = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return padded

    def start_processing_thread(self):
        """Arayüzün donmaması için işlemi ayrı bir kanalda (thread) çalıştırır."""
        self.btn_process.config(state=tk.DISABLED, text="İşleniyor...")
        thread = threading.Thread(target=self.run_process)
        thread.start()

    def run_process(self):
        """Ana işlem mantığı (Main.py'nin kalbi burası)"""
        try:
            self.log("-" * 40)
            self.log("1. Normalizasyon Başlatılıyor...")
            
            normalized_image = normalize_id_card(self.file_path, self.output_dir)

            if normalized_image is None:
                self.log("HATA: Normalizasyon başarısız. Kart bulunamadı.")
                self.reset_button()
                return

            self.log("✅ Normalizasyon başarılı.")
            self.log("2. Segmentasyon (Bölütleme) yapılıyor...")

            field_images = segment_fields_from_blobs(normalized_image, self.output_dir)

            if not field_images:
                self.log("HATA: Metin blokları bulunamadı.")
                self.reset_button()
                return

            self.log(f"✅ {len(field_images)} adet metin bloğu bulundu.")
            self.log("3. OCR Okuma ve Veri Çözümleme...")

            raw_ocr_lines = []
            
            for i, field_img in enumerate(field_images):
                # İyileştirme
                processed_img = self.preprocess_for_ocr(field_img)
                
                # Kayıt (Opsiyonel)
                cv2.imwrite(os.path.join(self.output_dir, f"gui_field_{i}.png"), processed_img)

                # Tesseract Ayarı
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist="ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789./- "'
                text = pytesseract.image_to_string(processed_img, config=custom_config)
                cleaned_text = text.replace('\n', ' ').strip()

                if len(cleaned_text) > 1:
                    raw_ocr_lines.append(cleaned_text)
                    self.log(f"   -> Okunan Ham Veri: {cleaned_text}")

            self.log("-" * 40)
            self.log("SONUÇLAR:")
            
            # Parser çağırma
            final_data = parse_id_card_data(raw_ocr_lines)
            
            # JSON formatında güzelce yazdır
            json_str = json.dumps(final_data, indent=4, ensure_ascii=False)
            self.log(json_str)
            
            # Sonuç dosyası
            json_path = os.path.join(self.output_dir, "result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            
            self.log(f"\nDosya kaydedildi: {json_path}")
            messagebox.showinfo("Başarılı", "İşlem tamamlandı!")

        except Exception as e:
            self.log(f"KRİTİK HATA: {str(e)}")
            messagebox.showerror("Hata", f"Bir hata oluştu:\n{str(e)}")
        
        finally:
            self.reset_button()

    def reset_button(self):
        self.btn_process.config(state=tk.NORMAL, text="BAŞLAT / OKU")

if __name__ == "__main__":
    root = tk.Tk()
    app = IDCardApp(root)
    root.mainloop()