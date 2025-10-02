import streamlit as st
import requests
import io
from PIL import Image

# =========================================================================
# 1. API Bilgileri ve Temel Ayarlar (Kendi Bilgilerinizle GÜNCELLEYİN)
# =========================================================================
# NOT: API HATASI çözülene kadar bu kodu kullanmayın. Direkt Drive indirme çözümüne geçin.
API_KEY = "rqSHwZoYtdlYlnMctixU" 
# Model Kimliği
FISTIK_MODEL_ID = "fistik-ojqcr/3"  
# Modelin güven eşiği
CONFIDENCE_THRESHOLD = 0.45 

# =========================================================================
# 2. Sayfa Yapılandırması ve Karşılama Ekranı
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
        color: #008000;
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
        "FORD FO": ("Fusarium Odaklı Hastalık", "Solma ve kahverengileşme görülebilir. **Öneri:** Hastalıklı bitki parçalarını uzaklaştırın."),
        "MYCOPT": ("Mycosphaerella Yaprak Hastalığı", "Küçük kahverengi lekeler ve erken yaprak dökümü. **Öneri:** Koruyucu bakır içerikli ilaçlar veya uygun fungisitler kullanın."),
        "SOKADE": ("Sokan ve Delen Zararlı Hasarı", "Böceklerin emgi veya delme sonucu oluşan hasar. **Öneri:** Zararlı türünü belirleyip uygun insektisit ile mücadele edin."),
        "FİZYOLOJİ": ("Çevresel Stres/Bozukluk", "Besin eksikliği veya ısı stresi. **Öneri:** Toprak analizi yapın, sulama ve gübreleme programını gözden geçirin."),
        "SONID": ("Tanımlanmamış Yaprak Hastalığı", "Modelin tespit ettiği bilinmeyen hastalık. **Öneri:** Uzman bir ziraat mühendisine başvurarak kesin teşhis koydurun.")
    }
    
    return info.get(class_name, ("Bilinmeyen Etiket", "Bu etiket için detaylı bilgi bulunmamaktadır."))


# =========================================================================
# 4. Streamlit Arayüzü ve Analiz Akışı
# =========================================================================

col1, col2 = st.columns([1, 1.5]) # Alanları ayır

uploaded_file = None

with col1:
    st.markdown("### 1️⃣ Görüntü Yükleme")
    st.info("⚠️ **DİKKAT:** Sadece .jpg, .jpeg, .png uzantılı, tek bir fıstık yaprağı resmi yükleyin.")
    uploaded_file = st.file_uploader(
        "Lütfen analiz edilecek resmi seçin:", 
        type=['jpg', 'jpeg', 'png']
    )
    
    # Butona basılma durumunu kontrol et
    if uploaded_file is not None:
        if st.button("AI Analizini Başlat 🚀", help="Modelin hastalıkları tespit etmesini sağlar.", type="primary"):
            st.session_state['run_analysis'] = True
        else:
            if 'run_analysis' not in st.session_state:
                st.session_state['run_analysis'] = False
    else:
        if 'run_analysis' in st.session_state:
            del st.session_state['run_analysis']


if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    with col2:
        st.markdown("### 2️⃣ Yüklenen Görüntü")
        st.image(image, use_container_width=True)

    
    if st.session_state.get('run_analysis', False):
        # Analiz başladı
        with st.spinner('Model analizi gerçekleştiriliyor... Lütfen bekleyin.'):
            st.markdown("---")
            st.markdown("### 3️⃣ Tespit Sonuçları")
            
            try:
                # API çağrısı
                results = get_inference_url(image_bytes)
                
                if "predictions" in results:
                    # Sadece güven eşiğini geçen tahminleri al
                    predictions_list = [p for p in results["predictions"] if p['confidence'] >= CONFIDENCE_THRESHOLD]
                    
                    if predictions_list:
                        st.success("✅ Olası Hastalıklar Tespit Edildi!")
                        
                        for prediction in predictions_list:
                            confidence = prediction['confidence']
                            class_name = prediction['class']
                            info_title, info_detail = get_disease_info(class_name)
                            
                            st.markdown(f"#### **{info_title}** ({class_name})")
                            st.progress(confidence) # Görsel güven çubuğu
                            
                            st.markdown(f"**AI Güven Puanı:** **`%{confidence * 100:.2f}`**")
                            st.markdown(f"**Açıklama:** {info_detail}")
                            st.markdown("---") 

                    else:
                         st.info(f"Model, **%{CONFIDENCE_THRESHOLD*100:.0f} güvenin üzerinde** bir hastalık/zararlı tespit edemedi. Görünüşe göre yaprak sağlıklı olabilir!")
                         st.balloons() 
                else:
                    st.info("Resimde belirgin bir hastalık/zararlı tespit edilemedi.")

            except requests.exceptions.HTTPError as e:
                # API'den gelen HTTP hatası. Bu, anahtarın geçersiz olduğunu gösterir.
                st.error(f"API Hatası (HTTP Error): Lütfen **yeni ve geçerli bir API Anahtarı** oluşturup koda yapıştırın.")
            except Exception as e:
                st.error("Analiz sırasında beklenmeyen bir hata oluştu.")
                st.exception(e)
