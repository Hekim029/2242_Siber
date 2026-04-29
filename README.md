# 2242_Siber
Askeri IoT Ağlarında Gerçek Zamanlı Saldırı Tespit ve
Siber Savunma Sistemi

Bu proje, Askeri IoBT (Muharebe Alanı Nesnelerinin İnterneti) ağlarında ve kısıtlı donanım kaynaklarına sahip uç cihazlarda (Edge) çalışmak üzere optimize edilmiş otonom bir saldırı tespit sistemidir. Sistem; ARP Zehirlenmesi (ARP Spoofing) ve Dağıtık Servis Dışı Bırakma (DDoS-SYN) saldırılarını, hafifletilmiş makine öğrenmesi algoritmaları (LightGBM/XGBoost) ile milisaniyeler içerisinde tespit eder.

Teknik Yığın ve Gereksinimler

Sistemin kararlı çalışması için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

Python 3.8+

Scapy: Gerçek zamanlı ağ trafiği dinleme ve paket analizi için.

LightGBM / XGBoost: Hafifletilmiş ve yüksek performanslı sınıflandırma modelleri için.

Pandas & Numpy: Veri işleme ve matris operasyonları için.

Scikit-learn: Model değerlendirme metrikleri ve ön işleme için.

Matplotlib & Seaborn: Performans grafiklerinin üretilmesi için.

Kurulum Adımları

Projeyi çalıştırmak için aşağıdaki adımları sırasıyla takip ediniz:

Depoyu Klonlayın:
git clone https://github.com/emryigitt/2242_Siber.git
cd 2242_Siber

Sanal Ortam Oluşturun ve Aktif Edin (Önerilir):
python -m venv venv
Windows için: .\venv\Scripts\activate
Linux/Mac için: source venv/bin/activate

Gerekli Kütüphaneleri Kurun:
pip install scapy lightgbm xgboost pandas numpy matplotlib seaborn scikit-learn

Ağ İzinlerini Yapılandırın:
main.py dosyası ağ trafiğini dinlediği için yönetici izinlerine (root/administrator) ihtiyaç duyar. Terminali yetkili olarak çalıştırdığınızdan emin olun.

Çalıştırma ve Kullanım Sırası

Proje modüllerini şu sırayla kullanabilirsiniz:

Eğitim (Training): Modelleri eğitmek ve hibrit mimariyi oluşturmak için:
python hybrid_train.py

Test ve Analiz: Eğitilen modelin başarısını ve hata payını görmek için:
python test_et.py

Görselleştirme: Karmaşıklık matrisi ve performans grafiklerini üretmek için:
python final_plots.py

Canlı Tespit (Deployment): Sistemi sahada/canlı ağda başlatmak için:
python main.py

Dosya Yapısı

data/: Eğitimde kullanılan ağ trafiği veri setleri.

models/: Eğitilmiş model dosyalarının saklandığı dizin.

main.py: Gerçek zamanlı ağ dinleme ve saldırı tespit mekanizmasını başlatan ana dosya.

hybrid_train.py: Edge-AI uyumlu model eğitim dosyası.

diagnostic_train.py: Modelin zayıf noktalarını tespit eden teşhis eğitimi.

test_et.py: Model test ve performans ölçüm betiği.

final_stress_test.py: Kısıtlı donanımlar için CPU/RAM ve gecikme stres testleri.

final_plots.py: Doğruluk oranları ve grafiksel çıktı üreten dosya.
