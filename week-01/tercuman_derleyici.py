"""
Temel Kavramlar: Tercuman vs Derleyici
=======================================

Bu dosya; tercuman (interpreter) ve derleyici (compiler) kavramlarini
aciklar ve Python'in nasil calistigini gosterir.
"""

# -----------------------------------------------------------------------
# TERCUMAN (INTERPRETER) NEDIR?
# -----------------------------------------------------------------------
# Tercuman, kaynak kodu satir satir okuyup, her satiri aninda makine
# diline cevirir ve hemen calistirir.
#
# Ozellikleri:
#   - Kodu onceden derlemeye gerek yoktur.
#   - Hatalar, ilgili satira gelindiginde aninda gorulur.
#   - Genellikle daha yavas calisir (her seferinde ceviri yapilir).
#   - Esnek ve test etmesi kolay.
#
# Ornekler: Python, Ruby, JavaScript (Node.js), PHP
#
# Python Tercuman Akisi:
#   Kaynak Kod (.py)
#       -> Python Tercumani (CPython)
#           -> Bytecode (.pyc)
#               -> Python Sanal Makinesi (PVM)
#                   -> Cikti

# -----------------------------------------------------------------------
# DERLEYICI (COMPILER) NEDIR?
# -----------------------------------------------------------------------
# Derleyici, kaynak kodun tamamini bir seferde alarak makine koduna
# (ya da ara dile) cevirir. Ortaya cikan program dogrudan calistirilir.
#
# Ozellikleri:
#   - Tum kodu once analiz eder, sonra calistirir.
#   - Hatalar derleme asamasinda yakalanir.
#   - Calisma zamani genellikle daha hizlidir.
#   - Cikti platforma ozgu binary dosyadir.
#
# Ornekler: C, C++, Go, Rust, Java (JVM bytecode'u uretir)

# -----------------------------------------------------------------------
# PYTHON NEREDE DURUYOR?
# -----------------------------------------------------------------------
# Python bir TERCUMANLI dildir, ancak arka planda bir ara adim vardir:
#   1. Kaynak kod (.py) -> bytecode (.pyc) haline getirilir.
#   2. Bytecode, Python Sanal Makinesi (PVM) tarafindan calistirilir.
# Bu yapiya "yorumlamali/derleme hibrid" de denilebilir.

# -----------------------------------------------------------------------
# CANLI ORNEK: Tercumanin satir satir calismasi
# -----------------------------------------------------------------------

print("=" * 55)
print("  TERCUMAN vs DERLEYICI - Canli Demo")
print("=" * 55)

# Satirlar teker teker islenir; hata olan satirda duraklar.
sayi1 = 10
sayi2 = 3

print(f"\n[1] Toplama   : {sayi1} + {sayi2} = {sayi1 + sayi2}")
print(f"[2] Cikarma   : {sayi1} - {sayi2} = {sayi1 - sayi2}")
print(f"[3] Carpma    : {sayi1} * {sayi2} = {sayi1 * sayi2}")
print(f"[4] Bolme     : {sayi1} / {sayi2} = {sayi1 / sayi2:.4f}")
print(f"[5] Tam Bolme : {sayi1} // {sayi2} = {sayi1 // sayi2}")
print(f"[6] Kalan     : {sayi1} % {sayi2} = {sayi1 % sayi2}")
print(f"[7] Us        : {sayi1} ** {sayi2} = {sayi1 ** sayi2}")

# -----------------------------------------------------------------------
# KARSILASTIRMA TABLOSU (ozet)
# -----------------------------------------------------------------------
print("\n" + "=" * 55)
print(f"{'Ozellik':<25} {'Tercuman':<15} {'Derleyici'}")
print("-" * 55)
karsilastirma = [
    ("Calisma sekli",   "Satir satir",  "Toptan"),
    ("Hata zamani",     "Calisma",      "Derleme"),
    ("Hiz",             "Yavas",        "Hizli"),
    ("Cikti",           "Yok (anlik)",  "Binary dosya"),
    ("Ornek diller",    "Python, JS",   "C, C++, Go"),
]
for satir in karsilastirma:
    print(f"{satir[0]:<25} {satir[1]:<15} {satir[2]}")
print("=" * 55)
