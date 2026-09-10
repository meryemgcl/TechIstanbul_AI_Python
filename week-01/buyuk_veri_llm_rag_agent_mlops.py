"""
Buyuk Veri (Big Data) Nedir?
==============================

Geleneksel veri isleme araclariyla yonetilemeyecek kadar buyuk,
hizli uretilen veya karmasik yapidaki veri kumelerine "Buyuk Veri" denir.

Buyuk Veri; 5V modeliyle tanimlanir:
    1. Volume   (Hacim)    : Verinin miktari (terabayt, petabayt)
    2. Velocity (Hiz)      : Verinin uretilme ve islenmesi hizi
    3. Variety  (Cesitlilik): Yapisal, yari yapisal, yapisiz veri
    4. Veracity (Dogruluk) : Verinin guvenirligi ve kalitesi
    5. Value    (Deger)    : Veriden elde edilen anlamin onemi
"""

# -----------------------------------------------------------------------
# LLM (Large Language Model) NEDIR?
# -----------------------------------------------------------------------
"""
LLM, milyarlarca parametreye sahip, buyuk metin veri setleri uzerinde
egitilmis derin ogrenme modelleridir.

- Metin uretir, soruları yanitlar, kod yazar, ozetler.
- Mimari: Genellikle Transformer tabanli.
- Ornekler: GPT-4, Gemini, Claude, LLaMA

Calisma Mantigı:
    Girdi Metni (Prompt)
        -> Tokenization (kelimeler parcalara bolunur)
            -> Embedding (sayisal vektore donusturulur)
                -> Transformer Katmanlari
                    -> Olasilik Dagilimi
                        -> Cikti Metni (Token)
"""

# -----------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation) NEDIR?
# -----------------------------------------------------------------------
"""
RAG, LLM'in yanit uretmeden once harici bir bilgi tabanindan
ilgili belgeleri arayip getirdigi bir mimaridir.

Neden RAG?
    - LLM'ler egitim kesim tarihinden sonrasini bilmez.
    - Ozel/kurumsal verileri dogrudan islememis olabilir.
    - RAG ile model guncel ve ozgun bilgiye erisebilir.

Akis:
    Kullanici Sorusu
        -> Embedding (vektore donustur)
            -> Vektor Veritabani Arama (ChromaDB, Pinecone vs.)
                -> En Alakali Belgeler Getirilir (Retrieve)
                    -> LLM'e [Belgeler + Soru] olarak verilir
                        -> Model Yaniti Uretir (Augmented Generation)
"""

# -----------------------------------------------------------------------
# AGENT (Yapay Zeka Ajani) NEDIR?
# -----------------------------------------------------------------------
"""
Agent; bir LLM'in sadece yanit vermekle kalmayip, hedef dogrultusunda
bagimsiz kararlar alarak araclar (tools) kullanan sisteme denir.

Bir Agent sunlari yapabilir:
    - Web arama
    - Kod calistirma
    - Dosya okuma/yazma
    - API cagirma
    - Baska Agentlari tetikleme (Multi-Agent)

ReAct Dongusu:
    Dusun (Reason) -> Hareket et (Act) -> Gozlemle (Observe) -> Tekrarla

Populer Framework'ler:
    LangChain, AutoGen, CrewAI, LlamaIndex
"""

# -----------------------------------------------------------------------
# MLOps NEDIR?
# -----------------------------------------------------------------------
"""
MLOps (Machine Learning Operations); ML modellerinin gelistirilmesi,
egitilmesi, dagitilmasi ve izlenmesi sureclerini otomatize eden
DevOps pratiklerinin ML'e uyarlanmisidir.

Temel MLOps Bilesenler:
    1. Veri Yonetimi     : Veri toplama, temizleme, versiyonlama
    2. Deney Takibi      : MLflow, Weights & Biases
    3. Model Egitimi     : Otomatik pipeline'lar
    4. Model Kaydi       : Model Registry
    5. Dagitim (Deploy)  : REST API, Docker, Kubernetes
    6. Izleme (Monitor)  : Model drift, veri kayması tespiti
    7. Geri Besleme      : Yeniden egitim tetikleme

MLOps Olmadan:
    Notebook -> Manuel test -> "Bende calisiyordu" -> Sorun

MLOps ile:
    CI/CD Pipeline -> Otomatik Test -> Otomatik Deploy -> Monitor
"""

# -----------------------------------------------------------------------
# CANLI PYTHON OZET
# -----------------------------------------------------------------------

kavramlar = {
    "Buyuk Veri": "Hacim, Hiz, Cesitlilik, Dogruluk, Deger (5V modeli)",
    "LLM":        "Buyuk dil modeli; metin anlama ve uretme",
    "RAG":        "LLM + Harici bilgi getirme = Dogru ve guncel yanit",
    "Agent":      "Kendi kararini veren, arac kullanan otonom LLM sistemi",
    "MLOps":      "ML modellerini uretimde yonetme ve otomatize etme",
}

print("=" * 60)
print("  TEMEL YAPAY ZEKA KAVRAMLARI - Ozet")
print("=" * 60)
for kavram, aciklama in kavramlar.items():
    print(f"\n[{kavram}]")
    print(f"  -> {aciklama}")
print("\n" + "=" * 60)

# Buyuk Veri 5V ornegi
print("\nBuyuk Veri - 5V Modeli:\n")
besv = [
    ("Volume   (Hacim)",      "Gunluk milyarlarca sosyal medya gonderisi"),
    ("Velocity (Hiz)",        "Borsa verileri milisaniyede guncellenir"),
    ("Variety  (Cesitlilik)", "Metin, goruntu, ses, log, JSON bir arada"),
    ("Veracity (Dogruluk)",   "Sahte haberler vs. gercek veriler"),
    ("Value    (Deger)",      "Veriden anlam cikararak is karari vermek"),
]
for v, ornek in besv:
    print(f"  {v:<28} : {ornek}")
print()
