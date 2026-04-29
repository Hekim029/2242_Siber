import pandas as pd
import numpy as np
import joblib
import glob
import os
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm

path = 'data/' 
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []

print(f"Toplam {len(all_files)} dosya taranıyor...")

for filename in tqdm(all_files, desc="Veriler Okunuyor", unit="dosya"):
    df = pd.read_csv(filename)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    dosya_adi = os.path.basename(filename).lower()
    if "benign" in dosya_adi:
        label = "NORMAL"
    elif "ddos" in dosya_adi:
        label = "DDOS_ATTACK"
    elif "spoofing" in dosya_adi or "arp" in dosya_adi:
        label = "SPOOFING_ATTACK"
    else:
        label = "OTHER_CYBER_ATTACK"
        
    df['Label'] = label
    li.append(df)

df_ultimate = pd.concat(li, axis=0, ignore_index=True)
print(f"\n[Veri birlestirildi]: Toplam {len(df_ultimate)} satır hazır.")

# Özellik Seçimi
onemli_sutunlar = ['Rate', 'IAT', 'Header_Length', 'rst_count', 'Duration', 
                   'syn_count', 'Tot size', 'Min', 'psh_flag_number', 'Max', 
                   'Protocol Type', 'HTTPS', 'Tot sum', 'Std', 'Number']

X = df_ultimate[onemli_sutunlar]
y = df_ultimate['Label']

# Final model parametreleri
ultimate_model = LGBMClassifier(
    n_estimators=2000, 
    learning_rate=0.03, 
    num_leaves=128, 
    max_depth=-1, 
    class_weight='balanced',
    n_jobs=-1,
    verbose=-1
)

# Çapraz doğrulama
print("\n[10-KATLI ÇAPRAZ DOĞRULAMA BAŞLATILIYOR]")
print("Bu aşama modelin 10 kez baştan eğitilmesini içerir, en uzun süren kısımdır.")

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_sonuclar = []

for i, (train_index, test_index) in enumerate(tqdm(kfold.split(X), total=10, desc="Testler Yapılıyor", unit="aşama")):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
    ultimate_model.fit(X_train_cv, y_train_cv)
    skor = ultimate_model.score(X_test_cv, y_test_cv)
    cv_sonuclar.append(skor)

# Sonuçlar
print("\n" + "="*60)
print("HYBRIDEFENDER ULTIMATE TEST SONUÇLARI")
print("="*60)
print(f"10 Farklı Testin Ortalaması: %{np.mean(cv_sonuclar)*100:.4f}")
print(f"Standart Sapma (Tutarlılık): {np.std(cv_sonuclar):.6f}")
print(f"En İyi Skor: %{np.max(cv_sonuclar)*100:.4f}")
print("="*60)
print("\nFinal modeli tüm veri setiyle mühürleniyor...")
ultimate_model.fit(X, y)
joblib.dump(ultimate_model, "models/hybridefender_ULTIMATE_CERTIFIED.pkl")
print("İşlem başarıyla tamamlandı.")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Grafikleri çizme işlemi
print("\nGrafikler ciziliyor...")

# Karmaşıklık matrisi grafiği
plt.figure(figsize=(12, 8))
y_pred = ultimate_model.predict(X)
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=ultimate_model.classes_, 
            yticklabels=ultimate_model.classes_)
plt.title('HybriDefender Final Karmaşıklık Matrisi')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.savefig('models/final_confusion_matrix.png')
plt.show()

# Özellik önem sırasını görselleştirme
importance = pd.DataFrame({'Etki': ultimate_model.feature_importances_, 'Parametre': onemli_sutunlar})
importance = importance.sort_values(by='Etki', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='Etki', y='Parametre', data=importance, palette='magma')
plt.title('Saldırı Tespitinde En Belirleyici Ağ Parametreleri')
plt.tight_layout()
plt.savefig('models/final_feature_importance.png')
plt.show()

print("\nGrafikler 'models/' klasörüne kaydedildi. İşlem tamam!")