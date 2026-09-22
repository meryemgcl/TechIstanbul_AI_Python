"""
Hafta 2 - Konu 3: Overfitting, Underfitting ve Basari Metrikleri
=================================================================
Overfitting: Model egitim verisini ezberler, yeni veriye genelleyemez.
Underfitting: Model cok basit kalir, egitim verisini bile dogru ogrenemez.
Bu dosyada farkli derinlikteki karar agaclari uzerinden bu kavramlar gosterilir.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

np.random.seed(7)
n = 300

# Veri: ogrenci basarisi
calisma  = np.random.uniform(0, 10, n)
katilim  = np.random.uniform(0, 100, n)
X = np.column_stack([calisma, katilim])
y = ((calisma * 8 + katilim * 0.5 + np.random.randn(n) * 5) > 60).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# ------------------------------------------------------------------
# 1. OVERFITTING vs UNDERFITTING KARSILASTIRMA
# ------------------------------------------------------------------
print("=" * 65)
print("  OVERFITTING vs UNDERFITTING - Karar Agaci Derinligi Etkisi")
print("=" * 65)
print(f"\n  {'Derinlik':<12} {'Train Acc':>10} {'Test Acc':>10} {'Fark':>10} {'Durum'}")
print(f"  {'-'*60}")

durumlar = []
for depth in [1, 2, 3, 5, 8, 15, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=7)
    dt.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, dt.predict(X_train))
    ts_acc = accuracy_score(y_test,  dt.predict(X_test))
    fark   = tr_acc - ts_acc
    derinlik_str = str(depth) if depth else "Sinırsız"
    if depth == 1:
        durum = "UNDERFITTING"
    elif fark > 0.12:
        durum = "OVERFITTING"
    else:
        durum = "Dengeli ✓"
    print(f"  {derinlik_str:<12} {tr_acc:>10.3f} {ts_acc:>10.3f} {fark:>10.3f}  {durum}")
    durumlar.append((depth, tr_acc, ts_acc))

print("""
[Yorum]
- Derinlik=1: Underfitting → Model cok basit, egitimi bile iyi ogrenemedik
- Derinlik=3-5: Dengeli → Hem eğitimde hem testte iyi sonuc
- Derinlik=Sinirsiz: Overfitting → Train %100, ama test kotu (ezber yapti)
""")

# ------------------------------------------------------------------
# 2. BASARI METRIKLERI - Derin Aciklama
# ------------------------------------------------------------------
print("=" * 65)
print("  BASARI METRIKLERI - Hasta Tespiti Ornegi")
print("=" * 65)

# En dengeli modeli kullanalim (depth=5)
model = DecisionTreeClassifier(max_depth=5, random_state=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print(f"""
Karmasiklik Matrisi:
  +--------------------+------------------+
  |                    | Tahmin: Pozitif  | Tahmin: Negatif
  +--------------------+------------------+------------------+
  | Gercek: Pozitif    |  TP = {tp:<8}   |  FN = {fn}
  | Gercek: Negatif    |  FP = {fp:<8}   |  TN = {tn}
  +--------------------+------------------+------------------+

  TP (True Positive)  = {tp}  → Hasta olani "hasta" dedik  ✓
  TN (True Negative)  = {tn}  → Sagliklıyı "saglikli" dedik  ✓
  FP (False Positive) = {fp}   → Saglıklıyı "hasta" zannettik  ✗ (gereksiz tedavi)
  FN (False Negative) = {fn}   → Hastayı kacirdik  ✗ (tehlikeli!)
""")

print(f"  Accuracy  = (TP+TN)/(TP+TN+FP+FN) = ({tp}+{tn})/({tp+tn+fp+fn}) = {acc:.3f}")
print(f"  Precision = TP/(TP+FP)            = {tp}/({tp}+{fp})             = {prec:.3f}")
print(f"  Recall    = TP/(TP+FN)            = {tp}/({tp}+{fn})             = {rec:.3f}")
print(f"  F1-Score  = 2*Prec*Rec/(Prec+Rec) =                              = {f1:.3f}")

print(f"""
[Neden sadece Accuracy yetmez?]
Diyelim ki hastalık toplumda %5 oranında görülüyor (n={len(y_test)} kisiden ~{int(len(y_test)*0.05)} hasta).
Model herkese "saglikli" dese accuracy = %95 cıkar → cok iyi görünür!
Ama aslinda sifir hasta bulmamistir. Bu yuzden hasta tespitinde
RECALL (duyarlilik) en kritik metriktir: hic hasta kacirma!

Kural:
  - Cok FN'den korkuyorsak (hasta kacirma, dolandiricilik) → RECALL'a bak
  - Cok FP'den korcuyorsak (yanlis alarm, gereksiz maliyet) → PRECISION'a bak
  - Her ikisi de onemli → F1-SCORE kullan
""")

# ------------------------------------------------------------------
# 3. MINI ML SENARYOSU - Ogrenci Basari Tahmini
# ------------------------------------------------------------------
print("=" * 65)
print("  MINI SENARYO: Ogrenci Basari Tahmini")
print("=" * 65)

print("""
Problem     : Bir ogrencinin dersi gecip gecmeyecegini tahmin et
Problem Tipi: Binary Classification (Gecti=1 / Kaldi=0)
Features    : calisma_saati, katilim_orani
Label       : gecti_mi (0 ya da 1)
Algoritma   : Decision Tree (max_depth=5)
""")

# Yeni ogrenciler icin tahmin
yeni_ogrenciler = np.array([
    [2.0, 40],   # Az calisan, dusuk katilim
    [7.5, 85],   # Cok calisan, yuksek katilim
    [5.0, 60],   # Orta duzey
    [1.0, 20],   # Cok az calisan
    [9.0, 95],   # Cok calisan, neredeyse hic devamsizligi yok
])

tahminler      = model.predict(yeni_ogrenciler)
olasiliklar    = model.predict_proba(yeni_ogrenciler)[:, 1]
etiket         = {0: "Kaldi", 1: "Gecti"}

print(f"  {'Ogrenci':<10} {'Calisma(saat)':<16} {'Katilim(%)':<12} {'Tahmin':<10} {'Gecme Olasiligi'}")
print(f"  {'-'*65}")
for i, (ogr, t, olasil) in enumerate(zip(yeni_ogrenciler, tahminler, olasiliklar)):
    print(f"  Ogrenci {i+1:<3} {ogr[0]:<16.1f} {ogr[1]:<12.0f} {etiket[t]:<10} %{olasil*100:.1f}")

print("\n" + "=" * 65)
print("  Hafta 2 - Tum Konular Islendi!")
print("=" * 65)
