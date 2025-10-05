import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter ve görsel ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NASA_Asteroid_Analyzer:
    def _init_(self, api_key="0uvRA28FaTBye6fon1dHoC4ahqRXLDJJh107wRRz"):
        self.api_key = api_key
        self.base_url = "https://api.nasa.gov/neo/rest/v1"
        self.asteroid_data = None
        
    def get_asteroid_data(self, page_count=10):
        """NASA'dan asteroid verilerini al"""
        print("🌌 NASA'dan asteroid verileri alınıyor...")
        
        all_asteroids = []
        for page in range(page_count):
            try:
                url = f"{self.base_url}/neo/browse?api_key={self.api_key}&page={page}&size=20"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if 'near_earth_objects' in data:
                    all_asteroids.extend(data['near_earth_objects'])
                    print(f"📄 Sayfa {page+1}/{page_count} tamamlandı - {len(data['near_earth_objects'])} asteroid")
                else:
                    print(f"⚠ Sayfa {page+1}'de veri bulunamadı")
                    
            except Exception as e:
                print(f"❌ Sayfa {page+1} alınırken hata: {e}")
                continue
        
        self.asteroid_data = all_asteroids
        print(f"✅ Toplam {len(all_asteroids)} asteroid verisi alındı!")
        return all_asteroids
    
    def clean_and_prepare_data(self):
        """Veriyi temizle ve analiz için hazırla"""
        if not self.asteroid_data:
            print("❌ Önce veri alınmalı!")
            return None
            
        print("🧹 Veri temizleniyor ve hazırlanıyor...")
        
        cleaned_data = []
        for asteroid in self.asteroid_data:
            try:
                # Asteroid temel bilgileri
                diameter_km = asteroid['estimated_diameter']['kilometers']
                avg_diameter = (diameter_km['estimated_diameter_min'] + diameter_km['estimated_diameter_max']) / 2
                
                # Yakın geçiş bilgileri
                close_approaches = asteroid.get('close_approach_data', [])
                latest_approach = close_approaches[0] if close_approaches else None
                
                cleaned_asteroid = {
                    'id': asteroid['id'],
                    'name': asteroid['name'],
                    'diameter_min_km': diameter_km['estimated_diameter_min'],
                    'diameter_max_km': diameter_km['estimated_diameter_max'],
                    'diameter_avg_km': avg_diameter,
                    'is_hazardous': asteroid['is_potentially_hazardous_asteroid'],
                    'absolute_magnitude': asteroid['absolute_magnitude_h'],
                    'nasa_jpl_url': asteroid['nasa_jpl_url'],
                    'close_approach_count': len(close_approaches)
                }
                
                # Yakın geçiş verisi varsa ekle
                if latest_approach:
                    cleaned_asteroid.update({
                        'last_approach_date': latest_approach['close_approach_date'],
                        'miss_distance_km': float(latest_approach['miss_distance']['kilometers']),
                        'relative_velocity_kmh': float(latest_approach['relative_velocity']['kilometers_per_hour'])
                    })
                
                cleaned_data.append(cleaned_asteroid)
                
            except Exception as e:
                print(f"⚠ Asteroid {asteroid.get('name', 'unknown')} işlenirken hata: {e}")
                continue
        
        self.df = pd.DataFrame(cleaned_data)
        print(f"✅ {len(self.df)} asteroid başarıyla temizlendi!")
        return self.df
    
    def create_visualizations(self):
        """Tüm görselleştirmeleri oluştur"""
        if not hasattr(self, 'df'):
            print("❌ Önce veri temizlenmeli!")
            return
        
        print("🎨 Görselleştirmeler oluşturuluyor...")
        
        # 1. Asteroid Boyut Dağılımı
        self._plot_size_distribution()
        
        # 2. Tehlike Durumu Dağılımı
        self._plot_hazardous_distribution()
        
        # 3. Büyük Asteroidler
        self._plot_largest_asteroids()
        
        # 4. Tehlikeli Asteroidler
        self._plot_hazardous_asteroids()
        
        # 5. Interaktif Scatter Plot
        self._create_interactive_plot()
        
        # 6. İstatistiksel Özet
        self._show_statistics()
        
        print("✅ Tüm görselleştirmeler tamamlandı!")
    
    def _plot_size_distribution(self):
        """Asteroid boyut dağılımını göster"""
        plt.figure(figsize=(12, 8))
        
        # Boyut dağılımı histogramı
        plt.subplot(2, 2, 1)
        plt.hist(self.df['diameter_avg_km'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Ortalama Çap (km)')
        plt.ylabel('Asteroid Sayısı')
        plt.title('Asteroid Boyut Dağılımı')
        plt.grid(True, alpha=0.3)
        
        # Kutu grafiği
        plt.subplot(2, 2, 2)
        plt.boxplot(self.df['diameter_avg_km'])
        plt.ylabel('Çap (km)')
        plt.title('Asteroid Boyutları - Kutu Grafiği')
        plt.grid(True, alpha=0.3)
        
        # Tehlikeli/Güvenli boyut karşılaştırması
        plt.subplot(2, 2, 3)
        hazardous = self.df[self.df['is_hazardous'] == True]['diameter_avg_km']
        non_hazardous = self.df[self.df['is_hazardous'] == False]['diameter_avg_km']
        
        plt.hist(hazardous, bins=20, alpha=0.7, label='Tehlikeli', color='red')
        plt.hist(non_hazardous, bins=20, alpha=0.7, label='Güvenli', color='green')
        plt.xlabel('Ortalama Çap (km)')
        plt.ylabel('Asteroid Sayısı')
        plt.title('Tehlikeli vs Güvenli Asteroid Boyutları')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Büyüklük kategorileri
        plt.subplot(2, 2, 4)
        size_categories = pd.cut(self.df['diameter_avg_km'], 
                               bins=[0, 0.1, 0.5, 1, 5, 100], 
                               labels=['Çok Küçük (<0.1km)', 'Küçük (0.1-0.5km)', 
                                      'Orta (0.5-1km)', 'Büyük (1-5km)', 'Çok Büyük (>5km)'])
        category_counts = size_categories.value_counts()
        
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        plt.title('Asteroid Büyüklük Kategorileri')
        
        plt.tight_layout()
        plt.savefig('asteroid_boyut_dagilimi.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_hazardous_distribution(self):
        """Tehlike durumu dağılımını göster"""
        plt.figure(figsize=(15, 5))
        
        # Pasta grafiği
        plt.subplot(1, 3, 1)
        hazardous_count = self.df['is_hazardous'].value_counts()
        colors = ['#2ecc71', '#e74c3c']  # Yeşil: Güvenli, Kırmızı: Tehlikeli
        plt.pie(hazardous_count.values, labels=['Güvenli', 'Tehlikeli'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Tehlike Durumu Dağılımı')
        
        # Çubuk grafik
        plt.subplot(1, 3, 2)
        hazardous_count.plot(kind='bar', color=colors)
        plt.title('Tehlikeli vs Güvenli Asteroid Sayısı')
        plt.xlabel('Tehlike Durumu')
        plt.ylabel('Asteroid Sayısı')
        plt.xticks(rotation=0)
        
        # Tehlikeli asteroidlerin boyut dağılımı
        plt.subplot(1, 3, 3)
        hazardous_df = self.df[self.df['is_hazardous'] == True]
        if len(hazardous_df) > 0:
            plt.hist(hazardous_df['diameter_avg_km'], bins=20, color='red', alpha=0.7, edgecolor='black')
            plt.xlabel('Ortalama Çap (km)')
            plt.ylabel('Asteroid Sayısı')
            plt.title('Tehlikeli Asteroid Boyut Dağılımı')
        else:
            plt.text(0.5, 0.5, 'Tehlikeli Asteroid Bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('tehlike_dagilimi.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_largest_asteroids(self):
        """En büyük 15 asteroidi göster"""
        plt.figure(figsize=(14, 8))
        
        # En büyük 15 asteroid
        largest_15 = self.df.nlargest(15, 'diameter_avg_km')
        
        # Yatay çubuk grafik
        plt.subplot(1, 2, 1)
        colors = ['red' if hazardous else 'blue' for hazardous in largest_15['is_hazardous']]
        y_pos = np.arange(len(largest_15))
        
        plt.barh(y_pos, largest_15['diameter_avg_km'], color=colors, alpha=0.7)
        plt.yticks(y_pos, largest_15['name'])
        plt.xlabel('Ortalama Çap (km)')
        plt.title('En Büyük 15 Asteroid')
        
        # Renk açıklaması
        plt.text(0.7, 0.95, 'Kırmızı: Tehlikeli\nMavi: Güvenli', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        # Büyük asteroidlerin tehlike durumu
        plt.subplot(1, 2, 2)
        large_asteroids = self.df[self.df['diameter_avg_km'] > 1]  # 1km'den büyükler
        if len(large_asteroids) > 0:
            large_hazardous = large_asteroids['is_hazardous'].value_counts()
            plt.pie(large_hazardous.values, labels=['Güvenli', 'Tehlikeli'], 
                   autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
            plt.title('Büyük Asteroidlerde (1km+) Tehlike Dağılımı')
        else:
            plt.text(0.5, 0.5, '1km+ Asteroid Bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('en_buyuk_asteroidler.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_hazardous_asteroids(self):
        """Tehlikeli asteroidleri detaylı göster"""
        hazardous_df = self.df[self.df['is_hazardous'] == True]
        
        if len(hazardous_df) == 0:
            print("⚠ Tehlikeli asteroid bulunamadı!")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Tehlikeli asteroid boyut dağılımı
        plt.subplot(2, 2, 1)
        plt.hist(hazardous_df['diameter_avg_km'], bins=15, color='red', alpha=0.7, edgecolor='black')
        plt.xlabel('Ortalama Çap (km)')
        plt.ylabel('Asteroid Sayısı')
        plt.title('Tehlikeli Asteroid Boyut Dağılımı')
        plt.grid(True, alpha=0.3)
        
        # En tehlikeli 10 asteroid (büyüklüğe göre)
        plt.subplot(2, 2, 2)
        top_hazardous = hazardous_df.nlargest(10, 'diameter_avg_km')
        y_pos = np.arange(len(top_hazardous))
        
        plt.barh(y_pos, top_hazardous['diameter_avg_km'], color='darkred', alpha=0.7)
        plt.yticks(y_pos, top_hazardous['name'])
        plt.xlabel('Ortalama Çap (km)')
        plt.title('En Büyük 10 Tehlikeli Asteroid')
        
        # Mutlak parlaklık vs boyut
        plt.subplot(2, 2, 3)
        plt.scatter(hazardous_df['absolute_magnitude'], hazardous_df['diameter_avg_km'], 
                   c='red', alpha=0.6, s=50)
        plt.xlabel('Mutlak Parlaklık')
        plt.ylabel('Ortalama Çap (km)')
        plt.title('Tehlikeli Asteroidler: Parlaklık vs Boyut')
        plt.grid(True, alpha=0.3)
        
        # Yakın geçiş sayısı
        plt.subplot(2, 2, 4)
        approach_counts = hazardous_df['close_approach_count'].value_counts().head(10)
        approach_counts.plot(kind='bar', color='orange', alpha=0.7)
        plt.xlabel('Yakın Geçiş Sayısı')
        plt.ylabel('Asteroid Sayısı')
        plt.title('Tehlikeli Asteroidlerin Yakın Geçiş Frekansı')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('tehlikeli_asteroidler.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_interactive_plot(self):
        """Interaktik Plotly grafiği oluştur"""
        try:
            # Scatter plot: Boyut vs Mutlak Parlaklık
            fig = px.scatter(self.df, 
                           x='absolute_magnitude', 
                           y='diameter_avg_km',
                           color='is_hazardous',
                           size='diameter_avg_km',
                           hover_data=['name'],
                           title='Asteroidler: Boyut vs Parlaklık',
                           labels={
                               'absolute_magnitude': 'Mutlak Parlaklık',
                               'diameter_avg_km': 'Ortalama Çap (km)',
                               'is_hazardous': 'Tehlikeli Mi?'
                           },
                           color_discrete_map={True: 'red', False: 'blue'})
            
            fig.write_html("interactive_asteroid_plot.html")
            print("✅ Interaktif grafik oluşturuldu: interactive_asteroid_plot.html")
            
        except Exception as e:
            print(f"⚠ Interaktif grafik oluşturulamadı: {e}")
    
    def _show_statistics(self):
        """İstatistiksel özet göster"""
        print("\n" + "="*50)
        print("📊 ASTEROID VERİ ANALİZİ - İSTATİSTİKSEL ÖZET")
        print("="*50)
        
        print(f"📈 Toplam Asteroid Sayısı: {len(self.df)}")
        print(f"🔴 Tehlikeli Asteroid Sayısı: {self.df['is_hazardous'].sum()}")
        print(f"🟢 Güvenli Asteroid Sayısı: {len(self.df) - self.df['is_hazardous'].sum()}")
        
        print(f"\n📏 Boyut İstatistikleri:")
        print(f"   • En Küçük Asteroid: {self.df['diameter_avg_km'].min():.4f} km")
        print(f"   • En Büyük Asteroid: {self.df['diameter_avg_km'].max():.2f} km")
        print(f"   • Ortalama Çap: {self.df['diameter_avg_km'].mean():.2f} km")
        print(f"   • Medyan Çap: {self.df['diameter_avg_km'].median():.2f} km")
        
        print(f"\n⚠ Tehlikeli Asteroid İstatistikleri:")
        hazardous = self.df[self.df['is_hazardous'] == True]
        if len(hazardous) > 0:
            print(f"   • Ortalama Çap: {hazardous['diameter_avg_km'].mean():.2f} km")
            print(f"   • En Büyük Tehlikeli: {hazardous['diameter_avg_km'].max():.2f} km")
            print(f"   • En Küçük Tehlikeli: {hazardous['diameter_avg_km'].min():.4f} km")
        
        print(f"\n🌍 Yakın Geçiş İstatistikleri:")
        print(f"   • Yakın Geçiş Yapan Asteroidler: {self.df['close_approach_count'].sum()}")
        print(f"   • Ortalama Yakın Geçiş Sayısı: {self.df['close_approach_count'].mean():.1f}")
        
        # Büyüklük kategorileri
        size_bins = [0, 0.1, 0.5, 1, 5, 100]
        size_labels = ['Çok Küçük (<0.1km)', 'Küçük (0.1-0.5km)', 'Orta (0.5-1km)', 'Büyük (1-5km)', 'Çok Büyük (>5km)']
        self.df['size_category'] = pd.cut(self.df['diameter_avg_km'], bins=size_bins, labels=size_labels)
        
        print(f"\n📊 Büyüklük Kategorileri:")
        for category in size_labels:
            count = len(self.df[self.df['size_category'] == category])
            percentage = (count / len(self.df)) * 100
            print(f"   • {category}: {count} asteroid (%{percentage:.1f})")
    
    def save_analysis_report(self):
        """Analiz raporunu kaydet"""
        if not hasattr(self, 'df'):
            print("❌ Önce analiz yapılmalı!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"asteroid_analysis_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("NASA ASTEROID VERİ ANALİZ RAPORU\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
            f.write(f"Analiz Edilen Asteroid Sayısı: {len(self.df)}\n\n")
            
            # Temel istatistikler
            f.write("TEMEL İSTATİSTİKLER:\n")
            f.write(f"- Tehlikeli Asteroid Sayısı: {self.df['is_hazardous'].sum()}\n")
            f.write(f"- Güvenli Asteroid Sayısı: {len(self.df) - self.df['is_hazardous'].sum()}\n")
            f.write(f"- En Büyük Asteroid: {self.df['diameter_avg_km'].max():.2f} km\n")
            f.write(f"- En Küçük Asteroid: {self.df['diameter_avg_km'].min():.4f} km\n")
            f.write(f"- Ortalama Çap: {self.df['diameter_avg_km'].mean():.2f} km\n\n")
            
            # En büyük 10 asteroid
            f.write("EN BÜYÜK 10 ASTEROID:\n")
            largest_10 = self.df.nlargest(10, 'diameter_avg_km')
            for i, (_, asteroid) in enumerate(largest_10.iterrows(), 1):
                hazard_status = "🔴 TEHLİKELİ" if asteroid['is_hazardous'] else "🟢 GÜVENLİ"
                f.write(f"{i:2d}. {asteroid['name']:20} - {asteroid['diameter_avg_km']:6.2f} km - {hazard_status}\n")
            
            f.write(f"\nRapor dosyası: {filename}\n")
        
        print(f"✅ Analiz raporu kaydedildi: {filename}")

def main():
    """Ana fonksiyon"""
    print("🚀 NASA ASTEROID VERİ GÖRSELLEŞTİRME PROJESİ")
    print("=" * 55)
    
    # NASA Asteroid Analyzer oluştur
    analyzer = NASA_Asteroid_Analyzer()
    
    # Veriyi al
    asteroid_data = analyzer.get_asteroid_data(page_count=10)
    
    if not asteroid_data:
        print("❌ Veri alınamadı! İnternet bağlantınızı kontrol edin.")
        return
    
    # Veriyi temizle
    cleaned_data = analyzer.clean_and_prepare_data()
    
    if cleaned_data is not None:
        # Görselleştirmeleri oluştur
        analyzer.create_visualizations()
        
        # Rapor kaydet
        analyzer.save_analysis_report()
        
        print("\n🎉 PROJE TAMAMLANDI!")
        print("📊 Oluşturulan dosyalar:")
        print("   • asteroid_boyut_dagilimi.png")
        print("   • tehlike_dagilimi.png") 
        print("   • en_buyuk_asteroidler.png")
        print("   • tehlikeli_asteroidler.png")
        print("   • interactive_asteroid_plot.html")
        print("   • asteroid_analysis_report_[tarih].txt")
        
        # Son bir özet göster
        hazardous_count = analyzer.df['is_hazardous'].sum()
        total_count = len(analyzer.df)
        print(f"\n📈 Özet: {total_count} asteroid analiz edildi, {hazardous_count} tanesi tehlikeli!")
        
    else:
        print("❌ Veri temizleme başarısız!")

if _name_ == "_main_":
    main()
