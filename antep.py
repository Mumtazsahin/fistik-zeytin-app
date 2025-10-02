import streamlit as st
import requests
import io
import os
from PIL import Image
from ultralytics import YOLO # Modelinizi yerel olarak çalıştırmak için gerekli
import zipfile
import shutil

# =========================================================================
# 1. GİTHUB LİMİTİNİ AŞMA VE MODEL YOLU (API KULLANILMIYOR)
# =========================================================================

MODEL_AĞIRLIKLARI_YOLU = "best.pt"

# SİZİN DOĞRUDAN İNDİRME LİNKİNİZ (Büyük dosya onay parametresi eklendi)
# Linki tarayıcınızda kontrol edin, indirme başlamalıdır.
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&confirm=t&id=1sry766_MjFuPuwneRn8Zdzpyy-mIlO-O" 

# Modelin güven eşiği (Hocanıza %45 güvenle tespit yaptığınızı gösterir)
CONFIDENCE_THRESHOLD = 0.45 


# =========================================================================
# 2. MODELİ İNDİRME VE YÜKLEME FONKSİYONLARI (Sadece Bir Kez Çalışır)
# =========================================================================

@st.cache_resource 
def load_model_from_disk():
    """Modelleri yerel olarak yükler, yoksa buluttan indirir."""
    
    if not os.path.exists(MODEL_AĞIRLIKLARI_YOLU):
        st.sidebar.warning("Model dosyası bulunamadı, buluttan indiriliyor...")
        
        # 1. Dosyayı doğrudan indirme isteği gönder
        try:
            response = requests.get(DOWNLOAD_URL, stream=True)
            response.raise_for_status() # Hata kontrolü
        except requests.exceptions.HTTPError as e:
            # Bu hata gelirse, ya link yanlış ya da Drive paylaşımında sorun var demektir.
            st.error("❌ Model indirme hatası. Lütfen Drive paylaşım linkini kontrol edin ve herkese açık olduğundan emin olun.")
            st.exception(e)
            return None
        
        # 2. İndirilen veriyi best.pt olarak kaydet
        with open(MODEL_AĞIRLIKLARI_YOLU, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.sidebar.success("✅ Model Başarıyla İndirildi!")

    # Modeli Ultralytics kütüphanesi ile yükle
    try:
        model = YOLO(MODEL_AĞIRLIKLARI_YOLU)
        st.sidebar.success("Model Başarıyla Yüklendi ve Kullanıma Hazır.")
        return model
    except Exception as e:
        st.error(f"❌ Model Yükleme Hatası: best.pt dosyasını kontrol edin. Hata: {e}")
        return None

# Uygulamanın başlangıcında modeli yükle
LOCAL_MODEL = load_model_from_disk()

# =========================================================================
# 3. YARDIMCI VE TESPİT FONKSİYONLARI
# =========================================================================

def run_local_inference(image_bytes, model, confidence_threshold):
    """Yerel model ağırlıklarını kullanarak tahmin yapar."""
    
    if model is None:
        return {"predictions": []}

    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(image_bytes)
    
    # YOLO predict fonksiyonu
    results = model.predict(source=temp_file_path, conf=confidence_threshold, verbose=False)
    
    # Tahmin sonuçlarını işleyerek JSON benzeri bir yapıya dönüştürün
    predictions = []
    if len(results) > 0:
        for box in results[0].boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            class_name = results[0].names[cls]
            
            predictions.append({
                "class": class_name,
                "confidence": conf,
            })
            
    # os.remove(temp_file_path) # Geçici dosyayı silme
    return {"predictions": predictions}


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
# 4. Streamlit Arayüzü ve Analiz Akışı (API ÇAĞRISI KALDIRILDI)
# =========================================================================

st.set_page_config(page_title="Antep Fıstığı Hastalık Tespit Sistemi", layout="wide", initial_sidebar_state="expanded") 

st.markdown("""
    <style> .big-font { font-size:36px !important; font-weight: bold; color: #008000; text-align: center;}
    .medium-font { font-size:20px !important; color: #555555; text-align: center; }
    </style>
    <p class='big-font'>🌱 Yapay Zeka Destekli Fıstık Sağlığı Analizi 🥜</p>
    <p class='medium-font'>Fıstık yapraklarındaki hastalık ve zararlıları anında, yüksek doğrulukla tespit edin.</p>
""", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1.5]) 
uploaded_file = None

with col1:
    st.markdown("### 1️⃣ Görüntü Yükleme")
    st.info("⚠️ **DİKKAT:** Sadece .jpg, .jpeg, .png uzantılı, tek bir fıstık yaprağı resmi yükleyin.")
    uploaded_file = st.file_uploader("Lütfen analiz edilecek resmi seçin:", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        if st.button("AI Analizini Başlat 🚀", help="Modelin hastalıkları tespit etmesini sağlar.", type="primary"):
            st.session_state['run_analysis'] = True
        else:
            if 'run_analysis' not in st.session_state: st.session_state['run_analysis'] = False
    else:
        if 'run_analysis' in st.session_state: del st.session_state['run_analysis']


if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    with col2:
        st.markdown("### 2️⃣ Yüklenen Görüntü")
        st.image(image, use_container_width=True)

    
    if st.session_state.get('run_analysis', False):
        if LOCAL_MODEL is None:
            st.error("Model yüklenemediği için analiz başlatılamadı. Lütfen Drive bağlantısını ve paylaşım ayarlarını kontrol edin.")
        else:
            # Analiz başladı
            with st.spinner('Model analizi gerçekleştiriliyor... Lütfen bekleyin.'):
                st.markdown("---")
                st.markdown("### 3️⃣ Tespit Sonuçları")
                
                try:
                    # YEREL MODEL ÇAĞRISI
                    results = run_local_inference(image_bytes, LOCAL_MODEL, CONFIDENCE_THRESHOLD)
                    
                    if "predictions" in results:
                        predictions_list = results["predictions"]
                        
                        if predictions_list:
                            st.success("✅ Olası Hastalıklar Tespit Edildi!")
                            
                            for prediction in predictions_list:
                                confidence = prediction['confidence']
                                class_name = prediction['class']
                                info_title, info_detail = get_disease_info(class_name)
                                
                                st.markdown(f"#### **{info_title}** ({class_name})")
                                st.progress(confidence)
                                st.markdown(f"**AI Güven Puanı:** **`%{confidence * 100:.2f}`**")
                                st.markdown(f"**Açıklama:** {info_detail}")
                                st.markdown("---") 

                        else:
                             st.info(f"Model, **%{CONFIDENCE_THRESHOLD*100:.0f} güvenin üzerinde** bir hastalık/zararlı tespit edemedi. Görünüşe göre yaprak sağlıklı olabilir!")
                             st.balloons() 
                    else:
                        st.info("Resimde belirgin bir hastalık/zararlı tespit edilemedi.")

                except Exception as e:
                    st.error("Analiz sırasında beklenmeyen bir hata oluştu.")
                    st.exception(e)
