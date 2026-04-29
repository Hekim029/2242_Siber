import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import time
import joblib

df_saldiri = pd.read_csv("data/ARP_Spoofing_train.pcap.csv")
df_normal = pd.read_csv("data/Benign_train.pcap.csv")

df_saldiri['Label'] = 1
df_normal['Label'] = 0

df_birlesik = pd.concat([df_saldiri, df_normal], ignore_index=True)

# 4. Sonuc
print("--- Birleştirilmiş Veri Seti Hazır ---")
print(f"Toplam Satır: {df_birlesik.shape[0]}")
print(f"Toplam Sütun: {df_birlesik.shape[1]}")
print("\nSaldırı (1) ve Normal (0) Dağılımı:")
print(df_birlesik['Label'].value_counts())


X = df_birlesik.drop(['Label', 'Weight'], axis=1)
y = df_birlesik['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Model Sonuçları ---")
print(classification_report(y_test, y_pred))

print("\nLightGBM modeli eğitiliyor, lütfen bekleyin...")
model_lgb = LGBMClassifier(is_unbalance=True)
model_lgb.fit(X_train, y_train)

y_pred_lgb = model_lgb.predict(X_test)
print("\n--- LightGBM Model Sonuçları ---")
print(classification_report(y_test, y_pred_lgb))


ozellik_onemi = pd.DataFrame({'Değer': model_lgb.feature_importances_, 'Özellik': X.columns})

en_onemliler = ozellik_onemi.sort_values(by='Değer', ascending=False).head(15)

print("\n--- Saldırıyı Tespit Etmede En Önemli 15 Özellik ---")
print(en_onemliler)

onemli_sutunlar = ['Rate', 'IAT', 'Header_Length', 'rst_count', 'Duration', 
                   'syn_count', 'Tot size', 'Min', 'psh_flag_number', 'Max', 
                   'Protocol Type', 'HTTPS', 'Tot sum', 'Std', 'Number']

X_hafif = df_birlesik[onemli_sutunlar]
y_hafif = df_birlesik['Label']

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hafif, y_hafif, test_size=0.2, random_state=42)

print("\nHafifletilmiş LightGBM modeli eğitiliyor...")
model_hafif = LGBMClassifier(is_unbalance=True)
model_hafif.fit(X_train_h, y_train_h)

y_pred_hafif = model_hafif.predict(X_test_h)
print("\n--- Hafifletilmiş Model Sonuçları (Sadece 15 Özellik) ---")
print(classification_report(y_test_h, y_pred_hafif))

print("\nHız testi yapılıyor...")

tek_paket = X_test_h.iloc[[0]]

baslangic_zamani = time.time()
tahmin = model_hafif.predict(tek_paket)
bitis_zamani = time.time()

gecikme_ms = (bitis_zamani - baslangic_zamani) * 1000

print(f"--- Hız Testi Sonucu ---")
print(f"1 paketin analiz süresi: {gecikme_ms:.4f} milisaniye (ms)")
if gecikme_ms < 30:
    print("BAŞARILI: 30 ms hedefinin altındayız!")
else:
    print("BAŞARISIZ: 30 ms hedefini aştık.")

model_dosya_adi = "models/hybridefender_hizli_model.pkl"
joblib.dump(model_hafif, model_dosya_adi)
print(f"\nModel başarıyla kaydedildi: {model_dosya_adi}")

yuklenen_model = joblib.load(model_dosya_adi)
print("Model dosyadan başarıyla okundu.")