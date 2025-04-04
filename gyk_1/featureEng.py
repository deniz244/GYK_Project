import pandas as pd

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    #negatif/anormal verileri temizleme
    def basic_cleaning(self):
        self.df = self.df[self.df["quantity"] > 0]
        self.df = self.df[self.df["unit_price"] > 0]
        self.df = self.df[self.df["discount"].between(0, 1)] # %0 ile %100 arasında indirim olmalı
        print("Anlamlı olmayan veri temizliği yapıldı.")

    def process_date_features(self):
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        self.df['year'] = self.df['order_date'].dt.year
        self.df['month'] = self.df['order_date'].dt.month
        self.df['year_month'] = self.df['order_date'].dt.to_period('M')
        self.df['season'] = self.df['month'] % 12 // 3 + 1
        self.df['season'] = self.df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
        print("Tarih özellikleri eklendi.")

    def process_product_features(self):
        self.df['discounted_unit_price'] = self.df['unit_price'] * (1 - self.df['discount'])
        self.df['total_quantity'] = self.df['quantity']
        print("Ürün özellikleri eklendi.")

    def process_customer_features(self):
        self.df['customer_segment'] = self.df['company_name'].apply(lambda x: 'Segment A' if x[0] < 'N' else 'Segment B')
        print("Müşteri segmentasyonu eklendi.")

    def handle_missing_values(self):
        missing_before = self.df.isnull().sum().sum()
        self.df.dropna(inplace=True)
        missing_after = self.df.isnull().sum().sum()
        print(f"Temizleme yapıldı. Eksik veri öncesi: {missing_before}, sonrası: {missing_after}")

    def get_dataframe(self):
        return self.df