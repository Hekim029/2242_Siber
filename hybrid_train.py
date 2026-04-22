import pandas as pd
import joblib
import glob
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Dinamik Veri Yükleme
path = 'data/' 
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []

for filename in all_files:
    df = pd.read_csv(filename)
    attack_name = os.path.basename(filename).replace('_train.pcap.csv', '').replace('.csv', '')
    
    # DİKKAT: Artık 0-1 değil, direkt ismini veriyoruz
    df['Label'] = attack_name
    li.append(df)
    print(f"Teşhis Listesine Eklendi: {attack_name}")

df_diag = pd.concat(li, axis=0, ignore_index=True)

# 2. Özellik Seçimi (Aynı 15 özellik - Hızımız değişmeyecek)
onemli_sutunlar = ['Rate', 'IAT', 'Header_Length', 'rst_count', 'Duration', 
                   'syn_count', 'Tot size', 'Min', 'psh_flag_number', 'Max', 
                   'Protocol Type', 'HTTPS', 'Tot sum', 'Std', 'Number']

X = df_diag[onemli_sutunlar]
y = df_diag['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Eğitimi (Çoklu Sınıflandırma)
print("\nSaldırı türleri teşhis ediliyor, lütfen bekleyin...")
model = LGBMClassifier(class_weight='balanced', n_estimators=300)
model.fit(X_train, y_train)

# 4. Sonuçlar
print("\n--- Çoklu Saldırı Teşhis Raporu ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Kaydetme
joblib.dump(model, "models/hybridefender_diagnostic_model.pkl")
print("\nTeşhis modeli kaydedildi!")