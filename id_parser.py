import re

def clean_ocr_text(text):
    """OCR'dan gelen metni temizler ve standartlaştırır."""
    return text.strip().upper()

def correct_tc_ocr(text):
    """TC numarasındaki yaygın OCR hatalarını düzeltir."""
    replacements = {
        'B': '8', 'S': '5', 'O': '0', 'D': '0', 
        'I': '1', 'L': '1', 'Z': '2', 'G': '6'
    }
    for char, number in replacements.items():
        text = text.replace(char, number)
    return text

def parse_id_card_data(text_list):
    """
    OCR satır listesini alır, anlamlı kimlik verisine dönüştürür.
    """
    extracted_data = {
        "tc_kimlik_no": None,
        "soyad": None,
        "ad": None,
        "dogum_tarihi": None,
        "son_gecerlilik": None,
        "cinsiyet": None,
        "seri_no": None,
        "uyruk": None 
    }

    # 1. TC KİMLİK NO BULMA
    tc_index = -1
    tc_pattern = re.compile(r'^\d{11}$')
    
    for i, text in enumerate(text_list):
        clean = clean_ocr_text(text)
        potential_tc = correct_tc_ocr(clean)
        if tc_pattern.match(potential_tc):
            extracted_data["tc_kimlik_no"] = potential_tc
            tc_index = i
            break
            
    # 2. AD VE SOYAD BULMA
    if tc_index != -1:
        candidates = []
        for i in range(tc_index + 1, len(text_list)):
            line = clean_ocr_text(text_list[i])
            if len(line) < 2: continue
            if any(x in line for x in ["SERI", "NO", "TUR", "CUMHURIYET", "REPUBLIC", "IDENTITY", "T.C."]): continue
            if "/" in line: continue
            if re.search(r'\d', line): continue
            candidates.append(line)
        
        if len(candidates) >= 1: extracted_data["soyad"] = candidates[0]
        if len(candidates) >= 2: extracted_data["ad"] = candidates[1]

    # 3. TARİHLERİ BULMA
    date_pattern = re.compile(r'\d{2}\.\d{2}\.\d{4}')
    all_dates = []
    for text in text_list:
        clean = clean_ocr_text(text).replace('O', '0').replace('o', '0')
        found = date_pattern.findall(clean)
        all_dates.extend(found)
    
    if all_dates:
        try:
            all_dates.sort(key=lambda x: int(x.split('.')[-1]))
            extracted_data["dogum_tarihi"] = all_dates[0]
            if len(all_dates) > 1:
                extracted_data["son_gecerlilik"] = all_dates[-1]
        except: pass

    # 4. CİNSİYET BULMA (E / M yapısını tam al)
    gender_pattern = re.compile(r'([EK]\s?/\s?[MF])')
    for text in text_list:
        clean = clean_ocr_text(text)
        match = gender_pattern.search(clean)
        if match:
            extracted_data["cinsiyet"] = match.group(0)
            break

    # 5. SERİ NO BULMA (KURALA GÖRE DÜZELTME)
    # Kural: 1 Harf + 2 Rakam + 1 Harf + 5 Rakam (Toplam 9 Karakter)
    # Örn: A32U63410
    
    # Katı Regex (Düzeltme sonrası kontrol için)
    strict_seri_pattern = re.compile(r'^[A-Z]\d{2}[A-Z]\d{5}$')
    
    # OCR Rakam Düzeltme Haritası (Harf okunan rakamlar)
    digit_fix_map = {'O': '0', 'B': '8', 'S': '5', 'I': '1', 'D': '0', 'Z': '2', 'G': '6'}
    # OCR Harf Düzeltme Haritası (Rakam okunan harfler)
    letter_fix_map = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}

    for text in text_list:
        # Boşlukları sil
        clean = clean_ocr_text(text).replace(" ", "")
        
        # Sadece 9 karakterli adaylara bak
        if len(clean) == 9:
            # String'i listeye çevir (Karakter bazlı düzeltme için)
            chars = list(clean)
            
            # --- POZİSYON BAZLI DÜZELTME ---
            
            # 1. HARF (Index 0) ve 3. HARF (Index 3)
            for i in [0, 3]:
                if chars[i].isdigit(): # Eğer rakam varsa harfe çevirmeyi dene
                    if chars[i] in letter_fix_map:
                        chars[i] = letter_fix_map[chars[i]]
            
            # 2. RAKAMLAR (Index 1,2 ve 4,5,6,7,8)
            for i in [1, 2, 4, 5, 6, 7, 8]:
                if chars[i].isalpha(): # Eğer harf varsa rakama çevirmeyi dene
                    if chars[i] in digit_fix_map:
                        chars[i] = digit_fix_map[chars[i]]
            
            # Düzeltilmiş metni birleştir
            candidate = "".join(chars)
            
            # Şimdi Regex ile katı kurala uyuyor mu?
            if strict_seri_pattern.match(candidate):
                extracted_data["seri_no"] = candidate
                break
            
    # 6. UYRUK BULMA (DİNAMİK)
    uyruk_pattern = re.compile(r'([A-Z\.]+)\s?/\s?([A-Z]{3})')
    
    for text in text_list:
        clean = clean_ocr_text(text)
        match = uyruk_pattern.search(clean)
        # Cinsiyet satırıyla karışmaması için kontrol
        if match and "E / M" not in clean and "K / F" not in clean:
            extracted_data["uyruk"] = match.group(0)
            break

    return extracted_data