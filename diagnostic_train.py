import pandas as pd
import numpy as np
import joblib
import glob
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Dinamik Veri Yükleme ve Ön İşleme
path = 'data/' 
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []

print("Veriler okunuyor ve temizleniyor...")

for filename in all_files:
    df = pd.read_csv(filename)
    dosya_adi = os.path.basename(filename)
    
    # --- TEMİZLİK ADIMLARI ---
    # 1. Sonsuz (inf) değerleri NaN yap ve sonra sil
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 2. Tekrarlanan verileri sil (Modelin ezberlemesini önler)
    df.drop_duplicates(inplace=True)
    
    # --- İSİMLERİ SADELEŞTİRME ---
    if "Benign" in dosya_adi:
        attack_name = "Normal_Trafik"
    elif "ARP_Spoofing" in dosya_adi:
        attack_name = "ARP_Spoofing"
    elif "DDoS-SYN" in dosya_adi:
        attack_name = "DDoS_Attack"
    else:
        attack_name = "Diger_Saldiri"
    
    df['Label'] = attack_name
    li.append(df)
    print(f"Temizlendi ve Eklendi: {attack_name} | Kalan Satır: {len(df)}")

# Tüm veriyi birleştir
df_diag = pd.concat(li, axis=0, ignore_index=True)

# 2. Özellik Seçimi
onemli_sutunlar = ['Rate', 'IAT', 'Header_Length', 'rst_count', 'Duration', 
                   'syn_count', 'Tot size', 'Min', 'psh_flag_number', 'Max', 
                   'Protocol Type', 'HTTPS', 'Tot sum', 'Std', 'Number']

X = df_diag[onemli_sutunlar]
y = df_diag['Label']

# 3. Eğitim ve Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model Eğitimi
print("\nTemizlenmiş veri ile teşhis modeli eğitiliyor...")
# class_weight='balanced' sayesinde az olan saldırı türlerine daha fazla önem verir
model = LGBMClassifier(class_weight='balanced', n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# 5. Sonuçlar
print("\n--- Final Teşhis Raporu (Temizlenmiş Veri) ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Kaydetme
joblib.dump(model, "models/hybridefender_final_diagnostic.pkl")
print("\nKusursuz model kaydedildi: models/hybridefender_final_diagnostic.pkl")



# 1. Karmaşıklık Matrisi (Confusion Matrix)
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('HybriDefender - Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig('models/confusion_matrix.png')
    print("Grafik kaydedildi: models/confusion_matrix.png")
    plt.show()

# 2. Özellik Önem Sırası (Feature Importance)
def plot_feature_importance(model, features):
    feature_imp = pd.DataFrame({'Değer': model.feature_importances_, 'Özellik': features})
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Değer", y="Özellik", data=feature_imp.sort_values(by="Değer", ascending=False))
    plt.title('Saldırı Tespitinde En Etkili Parametreler')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    print("Grafik kaydedildi: models/feature_importance.png")
    plt.show()

# Fonksiyonları Çağır
siniflar = sorted(y_test.unique())
plot_confusion_matrix(y_test, y_pred, siniflar)
plot_feature_importance(model, onemli_sutunlar)