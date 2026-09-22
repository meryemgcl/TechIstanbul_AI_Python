"""
Hafta 2 - Konu 1: ML Proje Yasam Dongusu
==========================================
Bir makine ogrenmesi projesi hangi adimlardan olusur?
Bu dosyada Titanic veri seti uzerinden adim adim gosterilmektedir.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------------
# ADIM 1: Problem Tanimi
# ------------------------------------------------------------------
# Titanic yolcularinin ozelliklerine bakarak
# hayatta kalip kalmayacaklarini tahmin edecegiz.
# Bu bir ikili siniflandirma (binary classification) problemidir.
# Label: 0 = Hayatta Kalamadi, 1 = Hayatta Kaldi

print("=" * 60)
print("  ML PROJE YASAM DONGUSU - Titanic Ornegi")
print("=" * 60)

# ------------------------------------------------------------------
# ADIM 2: Veri Toplama / Sentetik Veri Olusturma
# ------------------------------------------------------------------
# Gercek Titanic dataseti yerine benzer yapida sentetik veri kullaniyoruz.
# Sutunlar: [yas, cinsiyet(0=erkek,1=kadin), bilet_sinifi(1-3), kardes_sayisi]

np.random.seed(42)
n = 200

yas            = np.random.randint(5, 75, n)
cinsiyet       = np.random.randint(0, 2, n)       # 0=Erkek, 1=Kadin
bilet_sinifi   = np.random.randint(1, 4, n)       # 1=1.sinif, 3=3.sinif
kardes_sayisi  = np.random.randint(0, 5, n)

X = np.column_stack([yas, cinsiyet, bilet_sinifi, kardes_sayisi])

# Hayatta kalma: kadinlar ve 1.sinif yolcular icin olasiliği daha yuksek
hayatta_kaldi = (
    (cinsiyet == 1) * 0.45 +          # kadin olmak avantaj saglar
    (bilet_sinifi == 1) * 0.30 +      # 1.sinif avantaj saglar
    np.random.rand(n) * 0.25          # rastgelelik
)
y = (hayatta_kaldi > 0.50).astype(int)

print(f"\n[ADIM 2] Veri Seti Hazir")
print(f"  Toplam Ornek  : {n}")
print(f"  Hayatta Kaldi : {y.sum()} ({y.sum()/n*100:.1f}%)")
print(f"  Hayatta Kalamadi: {(y==0).sum()} ({(y==0).sum()/n*100:.1f}%)")

# ------------------------------------------------------------------
# ADIM 3: Veri Hazirlama - Train / Validation / Test Bolumu
# ------------------------------------------------------------------
# Once test setini ayir (%20), kalanini train+validation olarak kullan
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# Kalan verinin %25'i validation olur => toplam verinin %20'si
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42
)

print(f"\n[ADIM 3] Veri Bolme Sonuclari")
print(f"  Egitim (Train)        : {len(X_train)} ornek")
print(f"  Dogrulama (Validation): {len(X_val)} ornek")
print(f"  Test                  : {len(X_test)} ornek")

# Olceklendirme (Feature Scaling)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ------------------------------------------------------------------
# ADIM 4 & 5: Model Secimi ve Egitim
# ------------------------------------------------------------------
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print(f"\n[ADIM 4-5] Model Egitildi: {type(model).__name__}")

# ------------------------------------------------------------------
# ADIM 6: Degerlendirme
# ------------------------------------------------------------------
def metrikleri_goster(gercek, tahmin, set_adi):
    acc  = accuracy_score(gercek, tahmin)
    prec = precision_score(gercek, tahmin, zero_division=0)
    rec  = recall_score(gercek, tahmin, zero_division=0)
    f1   = f1_score(gercek, tahmin, zero_division=0)
    print(f"\n--- {set_adi} Seti Sonuclari ---")
    print(f"  Accuracy  (Dogruluk)  : {acc:.3f}")
    print(f"  Precision (Kesinlik)  : {prec:.3f}")
    print(f"  Recall    (Duyarlilik): {rec:.3f}")
    print(f"  F1-Score              : {f1:.3f}")

print("\n[ADIM 6] Model Degerlendirmesi")
metrikleri_goster(y_train, model.predict(X_train), "Egitim")
metrikleri_goster(y_val,   model.predict(X_val),   "Dogrulama")
metrikleri_goster(y_test,  model.predict(X_test),  "Test (Final)")

# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"\nKarmasiklik Matrisi (Test Seti):")
print(f"                  Tahmin: Kaldi  Tahmin: Kaldi Degil")
print(f"  Gercek: Kaldi       {cm[0][0]:^8}     {cm[0][1]:^8}")
print(f"  Gercek: Kalmadi     {cm[1][0]:^8}     {cm[1][1]:^8}")

print(f"\n  TP={cm[1][1]}, TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}")
print(f"\n  TN: Normal maili dogru 'normal' tahmin ettik")
print(f"  TP: Spami dogru 'spam' bulduk")
print(f"  FP: Normal maili yanlis 'spam' sandik (spam klasorune attik)")
print(f"  FN: Spami gozden kacirdik, gelen kutusuna dustu")

print("\n" + "=" * 60)
print("  ML Yasam Dongusu Tamamlandi!")
print("=" * 60)
