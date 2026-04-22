import pandas as pd
import joblib
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("Veriler ve kayıtlı model yükleniyor...")

# 1. Sadece veriyi okuyoruz (EĞİTİM YOK)
path = 'data/' 
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []

for filename in all_files:
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    
    dosya_adi = os.path.basename(filename).lower()
    if "benign" in dosya_adi: label = "NORMAL"
    elif "ddos" in dosya_adi: label = "DDOS_ATTACK"
    elif "spoofing" in dosya_adi or "arp" in dosya_adi: label = "SPOOFING_ATTACK"
    else: label = "OTHER_CYBER_ATTACK"
        
    df['Label'] = label
    li.append(df)

df_ultimate = pd.concat(li, axis=0, ignore_index=True)

onemli_sutunlar = ['Rate', 'IAT', 'Header_Length', 'rst_count', 'Duration', 
                   'syn_count', 'Tot size', 'Min', 'psh_flag_number', 'Max', 
                   'Protocol Type', 'HTTPS', 'Tot sum', 'Std', 'Number']

X = df_ultimate[onemli_sutunlar]
y = df_ultimate['Label']

# 2. Kaydettiğimiz o "mühürlü" modeli çağırıyoruz
model = joblib.load("models/hybridefender_ULTIMATE_CERTIFIED.pkl")

# 3. Hemen tahmin yapıp grafikleri çizdiriyoruz
print("Grafikler çiziliyor...")
y_pred = model.predict(X)

# --- MATRİS GRAFİĞİ ---
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('HybriDefender Final Karmaşıklık Matrisi')
plt.savefig('models/final_confusion_matrix.png')
plt.show()

# --- ÖNEM SIRASI GRAFİĞİ ---
importance = pd.DataFrame({'Etki': model.feature_importances_, 'Parametre': onemli_sutunlar})
importance = importance.sort_values(by='Etki', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='Etki', y='Parametre', data=importance, palette='viridis')
plt.title('Saldırı Tespitinde En Belirleyici Parametreler')
plt.savefig('models/final_feature_importance.png')
plt.show()