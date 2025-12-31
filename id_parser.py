import re

def clean_ocr_text(text):
    """OCR'dan gelen metni temizler ve standartlaştırır."""
    return text.strip().upper()

def _to_digits_mapped(text):
    """Harf/rakam karışık OCR çıktısını sık kullanılan hatalara göre rakama çevirir."""
    replacements = {
        'B': '8', 'S': '5', 'O': '0', 'D': '0',
        'I': '1', 'L': '1', 'Z': '2', 'G': '6'
    }
    out = []
    for ch in text:
        if ch.isdigit():
            out.append(ch)
        elif ch in replacements:
            out.append(replacements[ch])
        else:
            out.append(ch)
    return "".join(out)

def find_tc_anywhere(text_list):
    """Tüm satırları tarayıp 11 haneli sayı dizisi yakalar (iç içe de olsa)."""
    tc_pattern = re.compile(r'\d{11}')
    for raw in text_list:
        cleaned = clean_ocr_text(raw)
        mapped = _to_digits_mapped(cleaned)
        found = tc_pattern.findall(mapped)
        if found:
            return found[0]
    return None

def find_tc_with_index(text_list):
    """11 haneli TC'yi ve bulunduğu satır indeksini döndürür."""
    tc_pattern = re.compile(r'\d{11}')
    for idx, raw in enumerate(text_list):
        cleaned = clean_ocr_text(raw)
        mapped = _to_digits_mapped(cleaned)
        found = tc_pattern.findall(mapped)
        if found:
            return found[0], idx
    return None, -1

def find_names(text_list):
    """
    OCR satırlarından ad/soyad tahmini.
    Mantık: sadece harf içeren, etiket olmayan, en az 2 uzunlukta blokları sırayla al.
    """
    blacklist = {"SERI", "NO", "TUR", "CUMHURIYET", "REPUBLIC", "IDENTITY", "T.C.", "T C"}
    candidates = []
    for raw in text_list:
        line = clean_ocr_text(raw)
        if any(x in line for x in blacklist): 
            continue
        if "/" in line: 
            continue
        if re.search(r'\d', line): 
            continue
        # Harf ve boşluktan ibaret olanları al
        if re.fullmatch(r'[A-ZÇĞİÖŞÜ\s\.]+', line) and len(line) >= 2:
            # Nokta ayırıcılarını boşluk kabul et
            tokens = [t for t in re.split(r'[.\s]+', line) if t]
            candidates.extend(tokens)
    soyad = candidates[0] if len(candidates) >= 1 else None
    ad = candidates[1] if len(candidates) >= 2 else None
    return soyad, ad

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

    # 1. TC KİMLİK NO BULMA (her yerde ara, indeksle)
    tc_found, tc_idx = find_tc_with_index(text_list)
    extracted_data["tc_kimlik_no"] = tc_found
    
    # 2. AD VE SOYAD BULMA
    # Önce TC bulunduysa onun +2 ve +3 sonrası alanları önceliklendir
    soyad = ad = None
    if tc_idx != -1:
        def normalize_letters(raw):
            # Harf dışı karakterleri boşluğa çevirip sadeleştir
            line = clean_ocr_text(raw)
            line = re.sub(r'[^A-ZÇĞİÖŞÜ]', ' ', line)
            line = " ".join(line.split())
            return line if len(line.replace(" ", "")) >= 2 else None

        def pick_name_from(idx):
            if idx < 0 or idx >= len(text_list):
                return None
            return normalize_letters(text_list[idx])

        # TC'den 2 sonrakini soyad, 4 sonrakini ad olarak al (1-based field dizilimine göre)
        soyad = pick_name_from(tc_idx + 2)
        ad = pick_name_from(tc_idx + 4)
    
    # Eğer sıralı yakalayamadıysak önceki generik yönteme düş
    if not soyad or not ad:
        fallback_soyad, fallback_ad = find_names(text_list)
        soyad = soyad or fallback_soyad
        ad = ad or fallback_ad
    
    extracted_data["soyad"] = soyad
    extracted_data["ad"] = ad

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
    # Bazı OCR'larda 1 karakter eksik (8 uzunluk) gelebiliyor; bu nedenle opsiyonel harf ile de kontrol ediyoruz.
    
    strict_seri_pattern = re.compile(r'^[A-Z]\d{2}[A-Z]\d{5}$')
    relaxed_seri_pattern = re.compile(r'^[A-Z]\d{2}[A-Z]?\d{5}$')  # 8-9 uzunluklu, ortadaki harf opsiyonel
    
    digit_fix_map = {'O': '0', 'B': '8', 'S': '5', 'I': '1', 'D': '0', 'Z': '2', 'G': '6'}
    letter_fix_map = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}

    for text in text_list:
        clean = clean_ocr_text(text).replace(" ", "")
        if len(clean) < 8 or len(clean) > 9:
            continue
        
        chars = list(clean)
        
        # 1. HARF pozisyonları: index 0 ve 3 (varsa)
        for i in [0, 3]:
            if i < len(chars) and chars[i].isdigit():
                if chars[i] in letter_fix_map:
                    chars[i] = letter_fix_map[chars[i]]
        
        # 2. RAKAM pozisyonları: 1,2 ve 4..8 (varsa)
        for i in [1, 2, 4, 5, 6, 7, 8]:
            if i < len(chars) and chars[i].isalpha():
                if chars[i] in digit_fix_map:
                    chars[i] = digit_fix_map[chars[i]]
        
        candidate = "".join(chars)
        
        if strict_seri_pattern.match(candidate) or relaxed_seri_pattern.match(candidate):
            extracted_data["seri_no"] = candidate
            break
            
    # 6. UYRUK BULMA (T.C./TUR için agresif düzeltme)
    def normalize_slash_field(text):
        # Harf ve nokta dışını at, büyük harf bırak
        norm = clean_ocr_text(text)
        norm = re.sub(r'[^A-Z\./]', '', norm)
        return norm

    def map_side(token, side):
        # Solda T.C., sağda TUR için sık hatalar
        token = token.replace('.', '')
        if side == "left":
            repl = {'B': 'T', '8': 'T', 'D': 'C', 'O': 'C', 'Q': 'C', '0': 'C', 'G': 'C'}
        else:
            repl = {'0': 'O', 'O': 'U', 'B': 'U', '8': 'B', 'G': 'R', 'Q': 'R', 'D': 'R'}
        return "".join(repl.get(c, c) for c in token)

    def try_fix_uyruk(text):
        norm = normalize_slash_field(text)
        parts = [p for p in re.split(r'[\/]', norm) if p]
        if len(parts) >= 2:
            left = map_side(parts[0], "left")
            right = map_side(parts[1], "right")
            if left.startswith("TC") and right.startswith("TUR"):
                return "T.C./TUR"
            if left.startswith("TC") and right.startswith("TCR"):  # bazı OCR'lar R'yi ekleyebilir
                return "T.C./TUR"
        return None

    best = None
    for text in text_list:
        fixed = try_fix_uyruk(text)
        if fixed:
            best = fixed
            break
    if best is None:
        # Fallback: en çok UR/TUR içeren slash'li aday
        candidates = []
        for text in text_list:
            norm = normalize_slash_field(text)
            if "/" in norm and ("UR" in norm or "TUR" in norm):
                candidates.append((len(norm), norm))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best = candidates[0][1]

    extracted_data["uyruk"] = best

    return extracted_data