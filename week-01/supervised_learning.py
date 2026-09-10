"""
Machine Learning - Supervised Learning (Denetimli Ogrenme)
==========================================================
Konu:
    Bir ogrencinin gunluk calisma suresi ve derse katilim yuzdesi
    kullanilarak sinavi gecip gecemeyecegi tahmin edilmektedir.

Kullanilan Algoritma:
    Logistic Regression (Lojistik Regresyon)

Kutuphaneler:
    numpy, scikit-learn, matplotlib
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. Veri Seti
# ------------------------------------------------------------------
# Her satir bir ogrenciyi temsil eder.
# Sutun 1: Gunluk calisma saati
# Sutun 2: Derse katilim yuzdesi (0-100)
X = np.array([
    [1, 40], [2, 50], [3, 60], [4, 70],
    [5, 75], [6, 80], [7, 85], [8, 90],
    [2, 30], [1, 20], [3, 45], [4, 55],
    [6, 65], [7, 78], [8, 88], [9, 92],
    [1, 35], [2, 55], [5, 70], [6, 72],
])

# Etiketler: 0 -> Kaldi, 1 -> Gecti
y = np.array([0, 0, 0, 1, 1, 1, 1, 1,
              0, 0, 0, 0, 1, 1, 1, 1,
              0, 0, 1, 1])

# ------------------------------------------------------------------
# 2. Egitim / Test Bolumu
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------------------
# 3. Model Olusturma ve Egitim
# ------------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 4. Tahmin ve Degerlendirme
# ------------------------------------------------------------------
y_pred = model.predict(X_test)

print("=" * 50)
print("  SINAV TAHMIN SONUCLARI")
print("=" * 50)
print(f"Dogruluk Orani (Accuracy) : {accuracy_score(y_test, y_pred):.2f}")
print()
print("Karmasiklik Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print()
print("Siniflandirma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Kaldi", "Gecti"]))

# ------------------------------------------------------------------
# 5. Yeni Ogrenciler Icin Tahmin
# ------------------------------------------------------------------
yeni_ogrenciler = np.array([
    [3, 55],   # Ogrenci A: 3 saat calisma, %55 katilim
    [7, 85],   # Ogrenci B: 7 saat calisma, %85 katilim
    [1, 30],   # Ogrenci C: 1 saat calisma, %30 katilim
])

tahminler = model.predict(yeni_ogrenciler)
etiketler = {0: "Kaldi", 1: "Gecti"}

print("=" * 50)
print("  YENİ OGRENCI TAHMINLERI")
print("=" * 50)
for i, (ogr, tahmin) in enumerate(zip(yeni_ogrenciler, tahminler)):
    print(f"Ogrenci {i+1}: {ogr[0]} saat / %{ogr[1]} katilim  ->  {etiketler[tahmin]}")

# ------------------------------------------------------------------
# 6. Gorsellestirilmis Karar Siniri
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))

gecti = X[y == 1]
kaldi = X[y == 0]

plt.scatter(gecti[:, 0], gecti[:, 1], color="green", label="Gecti", marker="o", s=80)
plt.scatter(kaldi[:, 0], kaldi[:, 1], color="red",   label="Kaldi", marker="x", s=80)

# Karar siniri
x1_range = np.linspace(0, 10, 300)
# w0*x1 + w1*x2 + b = 0  ->  x2 = -(w0*x1 + b) / w1
w = model.coef_[0]
b = model.intercept_[0]
x2_boundary = -(w[0] * x1_range + b) / w[1]
plt.plot(x1_range, x2_boundary, color="blue", linestyle="--", label="Karar Siniri")

plt.xlabel("Gunluk Calisma Saati")
plt.ylabel("Derse Katilim (%)")
plt.title("Logistic Regression - Sinav Gecme Tahmini")
plt.legend()
plt.tight_layout()
plt.savefig("sinav_tahmin_grafik.png", dpi=150)
plt.show()
print("\nGrafik 'sinav_tahmin_grafik.png' olarak kaydedildi.")
