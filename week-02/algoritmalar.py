"""
Hafta 2 - Konu 2: Algoritmalar - Linear Reg, Decision Tree, Random Forest
==========================================================================
Dört temel ML algoritmasinin kiyaslamali olarak gosterimi.
Ayni veri seti uzerinde farkli modeller egitilip sonuclari karsilastirilir.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

np.random.seed(0)

# ===========================================================
# 1. LINEAR REGRESSION - Ev Fiyat Tahmini
# ===========================================================
print("=" * 60)
print("  1. LINEAR REGRESSION - Ev Fiyat Tahmini")
print("=" * 60)

# Ozellikler: [metrekare, oda sayisi, bina yasi]
n = 150
metrekare  = np.random.randint(60, 300, n)
oda_sayisi = np.random.randint(1, 6, n)
bina_yasi  = np.random.randint(0, 40, n)

# Fiyat formulu: metrekare * 3000 + oda_sayisi * 15000 - bina_yasi * 2000 + gurultu
fiyat = (metrekare * 3000 + oda_sayisi * 15000
         - bina_yasi * 2000 + np.random.randint(-10000, 10000, n))

X_ev = np.column_stack([metrekare, oda_sayisi, bina_yasi])
y_fiyat = fiyat

X_tr, X_ts, y_tr, y_ts = train_test_split(X_ev, y_fiyat, test_size=0.2, random_state=0)

lin_reg = LinearRegression()
lin_reg.fit(X_tr, y_tr)
y_pred_lr = lin_reg.predict(X_ts)
rmse_lr = np.sqrt(mean_squared_error(y_ts, y_pred_lr))

print(f"\nModel: LinearRegression")
print(f"  RMSE (Hata Payi) : {rmse_lr:,.0f} TL")
print(f"\nOrnek Tahminler:")
for i in range(3):
    print(f"  Ev {i+1} | {X_ts[i,0]}m2 / {X_ts[i,1]} oda / {X_ts[i,2]} yas"
          f"  ->  Gercek: {y_ts[i]:,.0f} TL  |  Tahmin: {y_pred_lr[i]:,.0f} TL")

print("""
[Aciklama]
Linear Regression, bagimsiz degiskenler (X) ile surekli bir sayisal
cikti (y) arasindaki dogrusal iliskiyi modelleyerek tahmin yapar.
Yani: y = a*x1 + b*x2 + ... + c (dogru denklemi birden fazla boyuta genisletilir)
""")

# ===========================================================
# 2. LOGISTIC REGRESSION - Spam Tespiti
# ===========================================================
print("=" * 60)
print("  2. LOGISTIC REGRESSION - Spam E-Posta Tespiti")
print("=" * 60)

# Ozellikler: [kotu_kelime_sayisi, buyuk_harf_orani, link_sayisi]
kotu_kelime  = np.random.randint(0, 20, n)
buyuk_harf   = np.random.uniform(0, 1, n)
link_sayisi  = np.random.randint(0, 10, n)

# Spam olasaligi: cok kotu kelime + cok link = spam
spam_olasilik = (kotu_kelime * 0.05 + link_sayisi * 0.08
                 + buyuk_harf * 0.2 + np.random.rand(n) * 0.2)
y_spam = (spam_olasilik > 0.55).astype(int)

X_spam = np.column_stack([kotu_kelime, buyuk_harf, link_sayisi])
X_tr2, X_ts2, y_tr2, y_ts2 = train_test_split(X_spam, y_spam, test_size=0.2, random_state=0)

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_tr2, y_tr2)
acc_log = accuracy_score(y_ts2, log_reg.predict(X_ts2))

print(f"\nModel: LogisticRegression")
print(f"  Accuracy: {acc_log:.3f}  ({acc_log*100:.1f}% dogru tahmin)")
print("""
[Aciklama]
Logistic Regression siniflandirma algoritmasidir. Ciktiyi 0-1 arasina
sikistiran sigmoid fonksiyonu kullanarak her ornegin belirli bir sinifa
ait olma olasiligini hesaplar.
""")

# ===========================================================
# 3. DECISION TREE - Kredi Onay
# ===========================================================
print("=" * 60)
print("  3. DECISION TREE - Kredi Onay Tahmini")
print("=" * 60)

# Ozellikler: [gelir(bin TL), borc_orani, kredi_gecmisi(0-10)]
gelir       = np.random.randint(10, 100, n)
borc_orani  = np.random.uniform(0, 1, n)
kredi_gec   = np.random.randint(0, 11, n)

onay_skoru  = gelir * 0.01 + kredi_gec * 0.07 - borc_orani * 0.5 + np.random.rand(n) * 0.2
y_kredi     = (onay_skoru > 0.55).astype(int)

X_kredi = np.column_stack([gelir, borc_orani, kredi_gec])
X_tr3, X_ts3, y_tr3, y_ts3 = train_test_split(X_kredi, y_kredi, test_size=0.2, random_state=0)

dt = DecisionTreeClassifier(max_depth=4, random_state=0)
dt.fit(X_tr3, y_tr3)
acc_dt = accuracy_score(y_ts3, dt.predict(X_ts3))

print(f"\nModel: DecisionTree (max_depth=4)")
print(f"  Accuracy: {acc_dt:.3f}  ({acc_dt*100:.1f}% dogru tahmin)")
print(f"\n  Ornek Karar Yolu (Ust Duzey):")
print(f"    Gelir > 50K mi?")
print(f"      EVET -> Kredi gecmisi >= 6 mi?")
print(f"               EVET -> ONAYlandi")
print(f"               HAYIR-> Borc orani < 0.3 mi? -> ...")
print(f"      HAYIR-> REDDEDILDI")
print("""
[Aciklama]
Decision Tree, veriyi ardisik evet/hayir sorulariyla aga benzeri dallara
boler. Her dal bir kurali, her yaprak ise bir sinif etiketini temsil eder.
Sezgisel ve aciklanabilirligi yuksek bir algoritmadır.
""")

# ===========================================================
# 4. RANDOM FOREST - Overfitting Karsilastirma
# ===========================================================
print("=" * 60)
print("  4. RANDOM FOREST - Overfitting Kiyaslamasi")
print("=" * 60)

# Decision Tree cok derin olunca overfitting yapar, Random Forest bunu azaltir
dt_deep = DecisionTreeClassifier(max_depth=None, random_state=0)  # sinir yok -> overfitting
rf      = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

dt_deep.fit(X_tr3, y_tr3)
rf.fit(X_tr3, y_tr3)

acc_dt_train = accuracy_score(y_tr3, dt_deep.predict(X_tr3))
acc_dt_test  = accuracy_score(y_ts3, dt_deep.predict(X_ts3))
acc_rf_train = accuracy_score(y_tr3, rf.predict(X_tr3))
acc_rf_test  = accuracy_score(y_ts3, rf.predict(X_ts3))

print(f"\n  {'Model':<30} {'Train Acc':>10} {'Test Acc':>10} {'Durum':>15}")
print(f"  {'-'*65}")
print(f"  {'DecisionTree (sinirsiz):':<30} {acc_dt_train:>9.3f} {acc_dt_test:>9.3f}  -> OVERFITTING!")
print(f"  {'RandomForest (100 agac):':<30} {acc_rf_train:>9.3f} {acc_rf_test:>9.3f}  -> Dengeli")

print("""
[Aciklama]
Random Forest, yuzlerce farkli karar agacini birlestiren bir topluluk
(ensemble) yontemidir. Her agac verinin farkli bir alt kumesini gorur,
sonuc tum agaclarin tahminlerinin cogunluguyla belirlenir. Bu yaklasim
overfitting'i onler ve daha tutarli tahminler uretir.
""")

# ===========================================================
# ALGORITMA KIYASLAMA OZETI
# ===========================================================
print("=" * 60)
print("  ALGORITMA KIYASLAMASI")
print("=" * 60)
kiyaslama = [
    ("Linear Regression",  "Surekli sayi (fiyat, sicaklik)", "Yuksek",  "Dusuk",    "Dusuk"),
    ("Logistic Regression","Iki sinif (spam/degil)",         "Yuksek",  "Orta",     "Orta"),
    ("Decision Tree",      "Her ikisi de",                   "Orta",    "Yuksek",   "Orta"),
    ("Random Forest",      "Her ikisi de",                   "Dusuk",   "Yuksek",   "Yuksek"),
]
print(f"\n  {'Algoritma':<22} {'Problem Tipi':<28} {'Acikl.':<8} {'Ovfit Riski':<14} {'Basari'}")
print(f"  {'-'*85}")
for row in kiyaslama:
    print(f"  {row[0]:<22} {row[1]:<28} {row[2]:<8} {row[3]:<14} {row[4]}")
print()
