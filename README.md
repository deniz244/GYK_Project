# GYK_Project

📊 Fonksiyonel Gereksinimler(Data Loader)
Northwind veritabanı kullanılarak sipariş, müşteri, ürün ve satış verileri çekilmiştir.
Veriler orders, order_details, products, customers tabloları arasında JOIN işlemleri ile ilişkilendirilmiştir.
Toplam satış tutarı (unit_price * quantity * (1 - discount)) formülüyle hesaplanmıştır.

⚙️ Kurulum
Projede PostgreSQL veritabanına bağlantı kurmak için aşağıdaki Python paketleri kullanılmaktadır:
"pip install pandas sqlalchemy psycopg2"

🧩 Veri Yükleme Modülü
Veritabanından verileri çekmek için DataLoader sınıfı oluşturulmuştur.
from data_loader import DataLoader
"loader = DataLoader(user="kullanici", password="şifre", host="localhost", db_name="northwind")
df = loader.load_data()"
Veritabanı bağlantısı kurulduktan sonra SQL sorgusu ile gerekli veriler pandas DataFrame olarak yüklenir.



🔍 Veri Ön İşleme (Feature Engineering)
Veri setine anlamlı ve kullanılabilir özellikler kazandırmak amacıyla aşağıdaki adımlar uygulanmıştır:

✔️ Temel Temizlik (Basic Cleaning)
Negatif veya anlamsız veriler temizlendi:
quantity > 0
unit_price > 0
discount değerleri %0 ile %100 arasında olacak şekilde filtrelendi.

🗓️ Tarih Özellikleri
order_date değişkeni kullanılarak şu yeni sütunlar eklendi:
year: Sipariş yılı
month: Sipariş ayı
year_month: Yıl-Ay (dönemsel analizler için)
season: Mevsim (Kış, İlkbahar, Yaz, Sonbahar)

📦 Ürün Özellikleri
discounted_unit_price: İndirimli ürün fiyatı hesaplandı.
total_quantity: Siparişteki toplam ürün miktarı olarak eklendi.

👥 Müşteri Özellikleri
customer_segment: Firma adının ilk harfine göre Segment A/B olarak sınıflandırıldı.
(Not: Bu kısım örnekleme amaçlı olup sonradan daha gelişmiş segmentasyon modelleriyle değiştirilebilir.)

❗ Eksik Verilerle Başa Çıkma
Tüm sütunlar eksik değerler açısından kontrol edilmiştir.
Eksik veriler varsa dropna() ile kaldırılmıştır.



🤖 Model Eğitimi ve Değerlendirme
Proje kapsamında, total_sales değişkenini tahmin etmek amacıyla çeşitli regresyon algoritmaları denenmiştir.

🛠️ Kullanılan Algoritmalar
Linear Regression
Decision Tree Regressor
K-Nearest Neighbors Regressor (KNN)
Random Forest Regressor

📌 Özellikler (Features)
Model eğitiminde kullanılan bağımsız değişkenler:
unit_price
quantity
discount
year
month
category_id

📊 Eğitim ve Test Ayrımı
Veri seti %80 eğitim, %20 test olacak şekilde train_test_split() fonksiyonu ile ayrılmıştır.

🔁 Ölçekleme
Sadece KNN modeli için özellikler StandardScaler ile normalize edilmiştir.

🧪 Değerlendirme Metrikleri
Modeller şu iki metrik ile değerlendirilmiştir:
R² (R-Kare Skoru): Tahmin başarısını ölçer.
RMSE (Root Mean Squared Error): Ortalama hata miktarını gösterir.

🏆 En İyi Modelin Kaydedilmesi
Eğitilen modeller arasından en yüksek R² skoruna sahip olan model otomatik olarak model.pkl dosyasına kaydedilir.
Eğer en iyi model KNN ise, ilgili StandardScaler nesnesi de ayrıca scaler.pkl olarak kaydedilir.



🔮 Tahmin Modülü (Model Inference)
Eğitilen modelin yeni verilerle tahmin yapabilmesi için ModelPredictor sınıfı geliştirilmiştir.

🔄 Yükleme Adımları
model.pkl: Eğitilmiş ve kaydedilmiş en iyi model dosyası yüklenir.
scaler.pkl: Eğer model KNN ise, normalize işlemi için scaler da yüklenir (isteğe bağlı).

🧠 Tahmin Fonksiyonu
predict() fonksiyonu, kullanıcıdan gelen veri ile model tahmini yapar.

✅ Girdi Formatı:
{
    "unit_price": 15.0,
    "quantity": 20,
    "discount": 0.1,
    "year": 2022,
    "month": 4,
    "category_id": 5
}


🔁 İşlem Adımları
1-Girdi veri DataFrame'e dönüştürülür.
2-Eğer bir scaler varsa (örn. KNN modelinde), veriler normalize edilir.
3-Model üzerinden tahmin gerçekleştirilir.
4-Sonuç tek bir değer (tahmini satış tutarı) olarak döndürülür.

🚀 API Servisi (FastAPI)
Bu proje, eğitilmiş satış tahmin modelini sunmak için FastAPI tabanlı bir RESTful API içerir. Kullanıcılar, ürün bilgilerine erişebilir, satış tahmini yapabilir ve modeli yeniden eğitebilirler.

📦 Kullanılan Paketler
FastAPI: API oluşturmak için
Pydantic: Veri doğrulama için
joblib: Model ve scaler yüklemek için

📌 Uygulama Başlatma
"uvicorn app.main:app --reload"

🔗 API Endpoint’leri:
1️⃣ GET /
Ana endpoint, API'nin çalıştığını test etmek için.
{
  "message": "API workss!! that Prediction of Sales"
}

2️⃣ POST /predict
Yeni bir satış tahmini yapılmasını sağlar.

🧾 Örnek Girdi:

{
  "unit_price": 10.5,
  "quantity": 25,
  "discount": 0.1,
  "year": 2022,
  "month": 5,
  "category_id": 3
}

✅ Yanıt:
{
  "predicted_sale_value": 235.75
}

3️⃣ GET /products
Tüm ürün bilgilerini getirir (veritabanından).

4️⃣ GET /sales_summary
Aylık satış özetlerini döndürür. (Toplam satış, yıl, ay bazında)

5️⃣ POST /retrain
Modeli yeniden eğitir ve günceller. Model dosyaları (model.pkl, scaler.pkl) yeniden oluşturulur.

✅ Yanıt:
{
  "message": "Model başarıyla yeniden eğitildi ve güncellendi."
}

Bu API'yi isterseniz Swagger UI üzerinden test edebilirsiniz:
📍 http://localhost:8000/docs



🧪 Ana Uygulama (main.py)
Bu betik, veri yükleme aşamasından başlayarak satış tahminine kadar tüm süreci çalıştırır. Yerel testler veya toplu işlemler için kullanılabilir.

⚙️ İşlem Adımları
Veri Yükleme
PostgreSQL veritabanına bağlanarak veriler DataLoader sınıfı ile yüklenir.

Özellik Mühendisliği
Yüklenen veriler FeatureEngineer sınıfı ile işlenir:
Tarih bilgileri ayrıştırılır
Ürün ve müşteri bilgileri düzenlenir
Eksik veriler temizlenir

Model Eğitimi
ModelTrainer sınıfı ile birden fazla regresyon modeli eğitilir ve karşılaştırılır.
En başarılı model otomatik olarak kaydedilir (model.pkl).
Eğer en iyi model KNN ise, veriyi dönüştüren scaler (scaler.pkl) da ayrıca kaydedilir.

Tahmin
ModelPredictor sınıfı ile model yüklenir ve örnek bir giriş verisi ile tahmin yapılır.

▶️ Nasıl Çalıştırılır?
Terminal veya komut satırında şu komutla çalıştırabilirsiniz:
"python main.py"


🧾 Örnek Çıktı

     unit_price  quantity  discount  year  month  category_id  total_sales
0         18.0        10      0.10  2022      5            1        162.0

LinearRegression: R2 = 0.8724, RMSE = 14.32
DecisionTree: R2 = 0.8541, RMSE = 15.21
KNN: R2 = 0.8123, RMSE = 17.80
RandomForest: R2 = 0.8915, RMSE = 13.01
En iyi model kaydedildi (RandomForest) R2: 0.8915 → 'model.pkl.'
Tahmin edilen satış: 187.45


📦 Gereksinimler
Projeyi çalıştırmak için aşağıdaki Python paketlerinin yüklü olması gerekmektedir. Gerekli tüm bağımlılıkları requirements.txt dosyasından kolayca yükleyebilirsiniz.

Kurulum
Proje bağımlılıklarını yüklemek için terminal veya komut satırına aşağıdaki komutu girin:
"pip install -r requirements.txt"








