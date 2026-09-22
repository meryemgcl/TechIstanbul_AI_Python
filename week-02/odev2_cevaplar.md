# Uygulamalı Yapay Zekâ Bootcamp - 2. Hafta Ödevi

## 1. ML Proje Yaşam Döngüsü
Bir makine öğrenmesi projesi genelde şu temel adımlardan oluşur:
- **Problem tanımı:** Çözmek istediğimiz sorunu ve ulaşmak istediğimiz hedefi net bir şekilde belirleriz.
- **Veri toplama:** Modelin öğrenmesi için ihtiyaç duyduğumuz ham verileri farklı veri tabanlarından veya kaynaklardan bir araya getiririz.
- **Veri hazırlama:** Eksik veya hatalı verileri temizler, veriyi modelin anlayabileceği matematiksel formata dönüştürürüz.
- **Model seçimi:** Çözmeye çalıştığımız probleme (örneğin sınıflandırma mı, regresyon mu?) en uygun algoritmayı seçeriz.
- **Eğitim:** Hazırladığımız veriyi algoritmaya vererek modelin verideki kalıpları öğrenmesini sağlarız.
- **Test/değerlendirme:** Modelin performansını, daha önce hiç görmediği bir veri seti üzerinde ölçer ve yeterince iyi olup olmadığına karar veririz.
- **Kullanım:** Başarılı olan modeli canlı sisteme entegre edip gerçek dünyada tahminler üretmesini sağlarız.

## 2. Dataset İnceleme
- **Dataset Adı:** Titanic: Machine Learning from Disaster (Kaggle)
- **Amacı:** Yolcuların yaş, cinsiyet, bilet sınıfı gibi özelliklerine bakarak Titanik kazasında hayatta kalıp kalmayacaklarını tahmin etmek.
- **Satır/Sütun:** Eğitim seti için yaklaşık 891 satır ve 12 sütun bulunuyor.
- **Tahmin Edilmek İstenen Alan (Label):** "Survived" (Hayatta kaldı mı?) alanı. 0 ölümü, 1 ise hayatta kalmayı temsil eder.

## 3. Train / Validation / Test
Veriyi tek parça halinde kullanmak yerine üçe böleriz, çünkü modelin ezber yapmasını engellemek isteriz.
- **Train (Eğitim) Seti:** Modelin kalıpları ve matematiksel kuralları öğrenmesi için kullanılan ana veri parçasıdır.
- **Validation (Doğrulama) Seti:** Eğitim sürerken modelin ayarlarını (hiperparametrelerini) iyileştirmek ve aşırı öğrenmeye (overfitting) gidip gitmediğini kontrol etmek için kullandığımız ara test verisidir.
- **Test Seti:** Model tamamen eğitildikten sonra, gerçek dünyadaki performansını görmek için sadece en son aşamada kullandığımız, modelin eğitimde kesinlikle görmediği veridir.

## 4. Algoritmaları Tanıma
- **Linear Regression:** Ev fiyatı tahmini veya sıcaklık tahmini gibi, sonucun sürekli sayısal bir değer olduğu regresyon problemlerinde kullanılır.
- **Logistic Regression:** Bir hastanın kanser olup olmadığı veya bir e-postanın spam olup olmadığı gibi iki veya daha fazla gruba ayırma (sınıflandırma) problemlerinde kullanılır.
- **Decision Tree:** Veriyi ardışık sorular sorarak ağaç dalları gibi parçalara bölerek karar verdiği için hem sınıflandırma hem de regresyon problemlerinde kullanılabilir.
- **Random Forest:** Birçok farklı karar ağacını birleştirerek daha güçlü ve hataya daha dayanıklı tahminler yapmak için hem regresyon hem de sınıflandırma problemlerinde kullanılır.

## 5. Overfitting ve Underfitting
- **Overfitting (Aşırı Öğrenme):** Modelin eğitim verisini o kadar fazla ezberlemesi ki, yeni ve farklı veriler gördüğünde doğru tahmin yapamamasıdır. 
  - *Benzetme:* Sınav sorularını cevaplarıyla birlikte ezberleyen bir öğrencinin, sınavda sorunun sadece rakamları değiştiğinde soruyu çözememesi.
- **Underfitting (Yetersiz Öğrenme):** Modelin çok basit kalması ve verinin içindeki temel yapıyı bile öğrenememesi durumudur. 
  - *Benzetme:* Matematik sınavına girecek bir öğrencinin sadece toplama işlemini çalışıp, çıkarma ve çarpmaya hiç bakmadan sınava girmesi.

## 6. Başarı Metrikleri
- **Accuracy (Doğruluk):** Yaptığımız toplam tahminlerin yüzde kaçının doğru olduğunu gösterir.
- **Precision (Kesinlik):** Pozitif olarak tahmin ettiklerimizin gerçekte ne kadarının gerçekten pozitif olduğunu söyler.
- **Recall (Duyarlılık):** Gerçekte pozitif olan durumların ne kadarını doğru bir şekilde yakalayabildiğimizi gösterir.
- **F1-Score:** Precision ve Recall değerlerinin harmonik ortalamasıdır, bu iki değer arasında dengeli bir metrik sunar.
- **Hastalık tespitinde neden sadece Accuracy yetmez?** 
Diyelim ki bir hastalık toplumda sadece %1 oranında görülüyor. Eğer model herkese "hasta değil" derse, Accuracy %99 çıkar. Sayısal olarak çok başarılı görünse de aslında hiçbir hastayı bulamamıştır. Bu gibi durumlarda hasta olanları gözden kaçırmamak çok daha kritiktir, bu yüzden Recall (duyarlılık) metriğine bakmak gerekir.

## 7. Confusion Matrix (Karmaşıklık Matrisi)
Modelin yaptığı doğru ve yanlış tahminlerin gerçek değerlerle karşılaştırıldığı bir tablodur. Basit bir spam e-posta tespit modeli üzerinden düşünürsek:
- **TP (True Positive - Doğru Pozitif):** Gerçekte spam olan bir e-postayı, modelin başarıyla "spam" olarak tahmin etmesi.
- **TN (True Negative - Doğru Negatif):** Normal, temiz bir e-postayı, modelin "spam değil" (normal) olarak doğru tahmin etmesi.
- **FP (False Positive - Yanlış Pozitif):** Normal bir e-postayı, modelin yanlışlıkla "spam" zannedip spam klasörüne atması.
- **FN (False Negative - Yanlış Negatif):** Gerçekte spam olan zararlı bir e-postanın, modelin gözünden kaçıp gelen kutusuna normal bir mailmiş gibi düşmesi.

## 8. Mini ML Senaryosu
- **Problem:** Öğrenci Başarı Tahmini (Öğrencinin dersi geçip geçemeyeceğini önceden tahmin etmek)
- **Problem Tipi:** Sınıflandırma (Classification - Geçti/Kaldı)
- **Olası Feature'lar (Özellikler):** Haftalık ders çalışma süresi (saat), canlı derslere katılım oranı (%), daha önceki dönem not ortalaması, ödev teslim yüzdesi.
- **Label (Etiket):** Ders durumu (1 = Geçti, 0 = Kaldı)
- **Kullanılabilecek Algoritma:** Logistic Regression veya Decision Tree (Karar Ağacı).

---

## Kaynaklar
- Kaggle Datasetleri: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
- Scikit-Learn Belgeleri: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Kendi ders notlarım ve araştırmalarım.
