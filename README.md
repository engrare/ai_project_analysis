# AI Proje Analizi
AIProjectDataSet içerisinde, senaryo gereği, bir kurumun geliştirdiği projelerin yıllara göre değişen farklı parametreleri 60 satırlık bir veri seti olarak tutulmuştur. Projede bu veri seti incelenerek en sorunlu ve en yatırım yapılabilir projeler bulunmaya çalışılmıştır. 

Kodun okunabilirliğini kolaylaştırmak amacıyla kod içerisinde bulunan syntax özellikleri, makine öğrenmesi algoritmaları, özellik mühendisliği kavramları ve performans metrikleri detaylı bir şekilde açıklanacaktır.

### **Kod Dokümantasyonu ve Teknik Analiz Raporu**

#### **1. Syntax Özellikleri**

Bu bölümde, projenin temel yapı taşlarını oluşturan kütüphaneler, sınıf yapısı ve fonksiyonların teknik işleyişi açıklanmıştır.

* **Kullanılan Kütüphaneler ve Amaçları:**
* `pandas`: Veri setini okumak (CSV), veri çerçeveleri (DataFrame) oluşturmak, sütun bazlı işlemler ve veri manipülasyonu (gruplama, birleştirme) yapmak için kullanılır.
* `scipy.stats.linregress`: Bilimsel hesaplama kütüphanesinden çağrılan bu fonksiyon, veriler arasındaki doğrusal ilişkiyi (trendi) hesaplayarak eğim (slope) değerini bulur.
* `sklearn (Scikit-learn)`: Makine öğrenmesi algoritmaları için kullanılır.
* `MinMaxScaler`: Verileri belirli bir aralığa (genellikle 0-1) sıkıştırarak algoritmaların (kümeleme vb.) ölçek farkından etkilenmesini önler.
* `AgglomerativeClustering`: Hiyerarşik kümeleme algoritmasını uygular.
* `IsolationForest`: Anomali tespiti için kullanılır.


* `matplotlib.pyplot` ve `seaborn`: Analiz sonuçlarını görselleştirmek (grafik çizimi) için kullanılır.
* `warnings`: Kodun çalışmasını etkilemeyen uyarı mesajlarını gizleyerek çıktı ekranının temiz kalmasını sağlar.


* **Class (Sınıf) Yapısının Kullanımı:**
Kod, `AdvancedAIProjectAnalyzer` adında bir sınıf (class) yapısı üzerine kurulmuştur. Bu yapı, kodun modüler, okunabilir ve yeniden kullanılabilir olmasını sağlar.
* `__init__`: Sınıf başlatıldığında çalışır, dosya yolunu ve boş veri değişkenlerini tanımlar.
* Sınıf içindeki metodlar (`load_data`, `feature_engineering`, vb.) verinin yüklenmesinden raporlanmasına kadar olan işlem boru hattını (pipeline) sırasıyla yönetir.


* **Import Edilen Fonksiyonların İşlevi:**
Örneğin `linregress` fonksiyonu, bir proje için yıllara göre verimlilik puanlarını alıp, bu puanların zamanla arttığını mı yoksa azaldığını mı gösteren "eğim" değerini hesaplar.

#### **2. Makine Öğrenmesi Algoritmaları**

Projede projeleri sınıflandırmak ve analiz etmek için üç temel yaklaşım kullanılmıştır:

* **Hiyerarşik Kümeleme (Agglomerative Clustering - Davranışa Göre Gruplandırma):**
* Veri noktalarını benzerliklerine göre gruplayan "aşağıdan yukarıya" bir yaklaşımdır.
* Kodda `n_clusters=3` parametresi ile projeler; performans ve trend özelliklerine göre 3 ana gruba ayrılır.
* Algoritma sonrası grupların ortalama puanlarına bakılarak, en yüksek skora sahip grup **"STAR (Yüksek Performans)"**, en düşük grup **"RISKY (Düşük Performans)"**, diğerleri ise **"STANDARD"** olarak etiketlenir.


* **İzolasyon Ormanı (Isolation Forest - Aykırı Değer Tespiti):**
* Normal verilerin yoğunlaştığı bölgelerden uzak kalan, "aykırı" (outlier) verileri tespit eder.
* Rastgele karar ağaçları oluşturarak çalışır; anomali olan veriler daha az sayıda bölünme ile izole edilebilir.
* Kodda `contamination=0.15` parametresi ile verilerin en aykırı %15'lik kısmının anomali adayı (örneğin; beklenmedik başarı veya başarısızlık) olduğu varsayılır. Sonuçta `-1` değeri anomaliyi temsil eder.


* **Ağırlıklı Puanlama (Weighted Scoring - Karar Destek Mekanizması):**
* Projeleri tek bir kritere göre değil, birden fazla kriterin önem derecesine göre sıralamak için kullanılır.
* Kod içerisindeki formül şu şekildedir:


* Bu formül, projenin gelecekteki potansiyeline (Trend) en yüksek ağırlığı verirken, mevcut verimlilik ve bütçe uyumunu da hesaba katar.



#### **3. Özellik Mühendisliği (Feature Engineering) Kavramları**

Ham verinin makine öğrenmesi modelleri için anlamlı hale getirilmesi sürecidir.

* **Parametrelerin Anlamları ve Dönüşümleri:**
* **Veri:** `Cost` (Maliyet), `Investment` (Yatırım), `Fraud` (Dolandırıcılık Önleme Başarısı), `CSAT` (Müşteri Memnuniyeti), `ProcessingTime` (İşlem Süresi).
* **Dönüşüm:** CSV dosyasındaki veriler genellikle metin (string) formatında ve ondalık ayracı virgül (`,`) ile gelir. Kod, `str.replace(',', '.')` işlemi ile virgülleri noktaya çevirir ve `pd.to_numeric` ile metni sayısal (float) veriye dönüştürür. Hatalı veya boş veriler `0` ile doldurulur.



#### **4. Performans Metrikleri**

Projelerin başarısını ölçmek için türetilen matematiksel göstergelerdir.

* **Bütçe Sapması Skoru (Budget Deviation):**
* Maliyetin, planlanan yatırımdan ne kadar saptığını mutlak değer olarak ölçer.
* **Formül:**


* Bu değerin düşük olması (0'a yakın), projenin bütçeye sadık kaldığını gösterir. Kodda puanlama yapılırken `1 - Budget Deviation` kullanılarak sapmanın az olması ödüllendirilir.


* **Verimlilik Skoru (Efficiency Score):**
* Projenin çıktılarını (Başarı ve Memnuniyet) girdisine (Maliyet) oranlar.
* **Formül:**


* Burada `Fraud` ve `CSAT` değerlerinin yüksek olması, `Cost` değerinin düşük olması skoru artırır. Çarpan (1.000.000), sayıyı daha okunabilir bir ölçeğe taşımak içindir.


* **Trend Eğimi (Trend Slope - Doğrusal Regresyon):**
* Bir projenin yıllar içindeki performans değişim yönünü ve hızını ifade eder.
* **Yöntem:** Verimlilik skorları () ve Yıllar () arasında  doğrusu çizilir.
* **Matematiksel Açıklama:** Burada ** (eğim)** değeri hesaplanır.
* Eğer  ise: Proje performansı yıllar geçtikçe **artmaktadır** (Pozitif Trend).
* Eğer  ise: Proje performansı **düşmektedir** (Negatif Trend).
* Eğimin büyüklüğü, değişimin hızını gösterir. Kodda bu değer `linregress(Years, Efficiency_Scores)[0]` ile elde edilir.
