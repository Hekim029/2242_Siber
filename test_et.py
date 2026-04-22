import joblib
import pandas as pd

# 1. Daha önce kaydettiğimiz "beyni" yüklüyoruz
model = joblib.load("models/hybridefender_hizli_model.pkl")

# 2. Test etmek için gerçek bir veri örneği oluşturalım 
# (Burada normalde internetten gelen anlık veri olur, biz şimdilik elle veriyoruz)
ornek_veri = pd.DataFrame([{
    'Rate': 500.0, 'IAT': 0.01, 'Header_Length': 128, 'rst_count': 0, 
    'Duration': 0.5, 'syn_count': 0, 'Tot size': 256, 'Min': 60, 
    'psh_flag_number': 0, 'Max': 1500, 'Protocol Type': 6, 'HTTPS': 1, 
    'Tot sum': 1000, 'Std': 0.5, 'Number': 10
}])

# 3. Tahmin yap
tahmin = model.predict(ornek_veri)

print("\n--- HybriDefender Canlı Analiz Sistemi ---")
if tahmin[0] == 1:
    print("⚠️ UYARI: Saldırı Tespit Edildi! (ARP Spoofing Şüphesi)")
else:
    print("✅ DURUM: Trafik Güvenli. İşlem Devam Ediyor.")