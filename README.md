# 🚀 2242_Siber: HybriDefender'a Hoş Geldiniz! 🚀

**Askeri IoBT Ağlarında Gerçek Zamanlı Saldırı Tespit ve Siber Savunma Sistemi**

Bu proje, Askeri IoBT (Muharebe Alanı Nesnelerinin İnterneti) ağlarında ve kısıtlı donanım kaynaklarına sahip uç cihazlarda (Edge) otonom çalışmak üzere tasarlandı! 🛡️ Sistemimiz; ARP Zehirlenmesi ve DDoS-SYN gibi sinsi saldırıları, LightGBM/XGBoost gibi hafif makine öğrenmesi algoritmalarıyla milisaniyeler içinde avlıyor. ⚡

---

## 🛠️ Teknik Yığın ve Gereksinimler
Sistemin fişek gibi çalışması için şunlar lazım:

* 🐍 **Python 3.8+**
* 📡 **Scapy:** Canlı ağ trafiğini dinleyip paketleri koklamak için.
* 🧠 **LightGBM / XGBoost:** Hafif ama ölümcül sınıflandırma modellerimiz.
* 📊 **Pandas & Numpy:** Veri işleme ve matris cambazlıkları.
* ⚙️ **Scikit-learn:** Model metrikleri ve ön işleme.
* 📈 **Matplotlib & Seaborn:** O havalı performans grafiklerini çizmek için.

---

## ⚙️ Kurulum Adımları (Hadi Başlayalım!)

Sanal Ortam Oluşturun (Ortalık karışmasın):

Bash
python -m venv venv
# Windows için:
.\venv\Scripts\activate
# Linux/Mac için:
source venv/bin/activate

3️⃣ Kütüphaneleri Yükleyin:

Bash
pip install scapy lightgbm xgboost pandas numpy matplotlib seaborn scikit-learn
4️⃣ İzinleri Verin:
main.py ağı dinleyeceği için "Yönetici (Root/Admin)" izinlerine bayılır. Terminali yetkili açmayı unutmayın! 👮‍♂️

🚀 Çalıştırma Sırası
Eğitim Zamanı 🏋️‍♂️: Modelleri eğitip kas yapmak için:

python hybrid_train.py
Test ve Analiz 🧪: Başarımızı ve hatalarımızı görmek için:

python test_et.py
Grafikler 📉: Sonuçları havalı görsellere dökmek için:

python final_plots.py
Canlı Sahaya Çıkış ⚔️: Nöbeti başlatmak için:

Bash
python main.py
📂 Dosya Yapımız (Kim Nerede?)

📁 data/ : Eğitimde kullandığımız veri setlerimiz.

📁 models/ : Eğitilmiş canavar gibi modellerimiz burada yatar.

📄 main.py : Ağın nöbetçisi, ana dosyamız.

📄 hybrid_train.py : Edge-AI uyumlu model antrenörümüz.

📄 diagnostic_train.py : Modelin zayıf noktalarını bulan doktorumuz.

📄 test_et.py : Performans ölçümcümüz.

📄 final_stress_test.py : Sistemi donanımsal olarak terleten stres testimiz.

📄 final_plots.py : Görsel sanat yönetmenimiz.
