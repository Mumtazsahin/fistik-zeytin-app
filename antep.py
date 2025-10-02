import streamlit as st
import requests
import io
from PIL import Image

# =========================================================================
# 1. API Bilgileri ve Temel Ayarlar (Kendi Bilgilerinizle GÜNCELLEYİN)
# =========================================================================
# Roboflow Ayarlar -> API Anahtarları sayfasından aldığınız KİŞİSEL API Anahtarınız.
API_KEY = "rqSHwZoYtdlYlnMctixU" 
# Modeller -> fıstık 3 -> Kimlik kısmından aldığınız ID. (Örn: fistik-ojqcr/3)
FISTIK_MODEL_ID = "fistik-ojqcr/3" 
# Modelin tespit ettiği güvenin en az %45 olması durumunda göster.
CONFIDENCE_THRESHOLD = 0.45 

# =========================================================================
# 2. Sayfa Yapılandırması ve Karşılama Ekranı (Profesyonel Görünüm)
# =========================================================================
st.set_page_config(
    page_title="Antep Fıstığı Hastalık Tespit Sistemi",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# Yan Menü (Sidebar)
st.sidebar.markdown("# 🧠 AI Model Bilgisi")
st.sidebar.markdown("Bu sistem, Roboflow'da eğitilmiş **YOLOv8** tabanlı Nesne Algılama teknolojisiyle çalışır.")
st.sidebar.markdown(f"**Aktif Model Sürümü:** `{FISTIK_MODEL_ID}`")
st.sidebar.markdown(f"**Minimum Güven Eşiği:** **%{CONFIDENCE_THRESHOLD*100:.0f}**")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Mümtazşahin. Tüm hakları saklıdır.")


# Ana Karşılama Başlığı (HTML ile ortalanmış ve vurgulanmış)
st.markdown("""
    <style>
    .big-font {
        font-size:36px !important;
        font-weight: bold;
        color: #008000; /* Koyu yeşil */
        text-align: center;
    }
    .medium-font {
        font-size:20px !important;
        color: #555555;
        text-align: center;
    }
    </style>
    <p class='big-font'>🌱 Yapay Zeka Destekli Fıstık Sağlığı Analizi 🥜</p>
    <p class='medium-font'>Fıstık yapraklarındaki hastalık ve zararlıları anında, yüksek doğrulukla tespit edin.</p>
""", unsafe_allow_html=True)
st.markdown("---")


# =========================================================================
# 3. Yardımcı Fonksiyonlar ve API Bağlantısı
# =========================================================================

def get_inference_url(image_bytes):
    """Görüntüyü Fıstık modeline gönderir ve tahminleri JSON olarak alır."""
    
    # Nesne Algılama API endpoint'i
    url = f"https://detect.roboflow.com/{FISTIK_MODEL_ID}?api_key={API_KEY}"
    
    response = requests.post(
        url,
        data=image_bytes,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    response.raise_for_status() # HTTP hatalarını yakala
    return response.json()

@st.cache_data
def get_disease_info(class_name):
    """Hastalık adı verilen bilgi kartını ve önerileri döndürür."""
    
    info = {
        "PHYPSO": ("Yaprak Lekesi (Phyllosticta)", "Yaprakta koyu dairesel noktalarla karakterizedir. **Öneri:** Hızlı mantar ilacı uygulaması ve iyi hava sirkülasyonu sağlayın."),
        "FORD FO