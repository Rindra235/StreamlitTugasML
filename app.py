# app.py
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import time
import platform 
import pathlib
from fastai.vision.all import * 
import torch.nn as nn

# --- Konfigurasi Halaman & Gaya CSS ---
st.set_page_config(
    page_title="Car Grader - AI Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Modern dengan Tema Hijau/Biru Cerah
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #00C9FF 0%, #92FE9D 100%);
    }
    
    .css-17eq0hr {
        background: linear-gradient(180deg, #00C9FF 0%, #92FE9D 100%);
    }
    
    /* Main Content Background */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Header Gradient */
    .hero-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
    }
    
    .hero-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-header p {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Modern Card Styling */
    .modern-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .result-card.success {
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%);
        border-left-color: #4ecdc4;
    }
    
    .result-card.warning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        border-left-color: #fdcb6e;
    }
    
    .result-card.danger {
        background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
        border-left-color: #e84393;
    }
    
    /* Animated Elements */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* CTA Button */
    .cta-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        color: white;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        text-decoration: none;
        display: inline-block;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(255,107,107,0.4);
        transition: all 0.3s ease;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(255,107,107,0.6);
        color: white;
        text-decoration: none;
    }
    
    /* Sidebar Logo */
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    /* Progress Bar Custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom Metrics */
    .metric-card {
        background: linear-gradient(135deg, #e3ffe7 0%, #d9e7ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* File Uploader Enhancement */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Map Enhancement */
    .deck-map {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# DEFINISI KELAS MODEL

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Lapisan Konvolusi (Tubuh Model)
        self.body = nn.Sequential(
            # Block 1: Conv(32) -> BN -> ReLU -> MaxPool
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            # Block 2: Conv(64) -> BN -> ReLU -> MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            # Block 3: Conv(128) -> BN -> ReLU -> MaxPool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        
        # Lapisan Fully Connected (Kepala Model)
        # Ukuran input setelah flatten: 128 channels * 18 * 18 = 41472 (untuk input 150x150)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
        
# --- Fungsi & Model Loading ---

plt_system = platform.system()
if plt_system == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

@st.cache_resource
def load_damage_model():
    """Memuat model klasifikasi kerusakan fast.ai."""
    try:
        # Menggunakan load_learner dari fast.ai untuk file .pkl
        learn = load_learner('custom_cnn_model.pkl')
        return learn
    except Exception as e:
        # Pesan error disesuaikan
        st.error(f"Error: Model file 'custom_cnn_model.pkl' not found or corrupted. {e}")
        return None

learn = load_damage_model()

# Konfigurasi
CLASS_NAMES = ['01-minor', '02-moderate', '03-severe']
SEVERITY_MAPPING = {
    '01-minor': 'Minor Damage',
    '02-moderate': 'Moderate Damage',
    '03-severe': 'Severe Damage'
}
IMG_WIDTH, IMG_HEIGHT = 150, 150

def make_prediction(image_pil):
    """Fungsi untuk melakukan prediksi menggunakan model fast.ai."""
    if learn is not None:
        try:
            # Prediksi dengan fast.ai mengembalikan (kelas, index, probabilitas)
            pred_class, pred_idx, outputs = learn.predict(image_pil)
            # Kita butuh array probabilitas untuk fungsi display_results
            # Ubah tensor output menjadi numpy array dan tambahkan dimensi batch
            prediction_array = np.expand_dims(outputs.numpy(), axis=0)
            return prediction_array
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            return None
    else:
        st.warning("Model tidak berhasil dimuat, prediksi tidak dapat dilakukan.")
        return None
    
def display_results(image, prediction):
    """Fungsi untuk menampilkan hasil prediksi dengan desain modern."""
    predicted_class_idx = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    confidence = np.max(prediction)
    severity = SEVERITY_MAPPING.get(predicted_class_name, "Unknown")

    # Determine card class based on severity
    card_class = "success" if predicted_class_name == '01-minor' else \
                "warning" if predicted_class_name == '02-moderate' else "danger"
    
    st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
    st.markdown("<h3>🎯 Assessment Results</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if predicted_class_name == '01-minor':
            st.success(f"**Predicted Severity: {severity}**", icon="✅")
        elif predicted_class_name == '02-moderate':
            st.warning(f"**Predicted Severity: {severity}**", icon="⚠️")
        else:
            st.error(f"**Predicted Severity: {severity}**", icon="🚨")
    
    with col2:
        st.markdown(f"<div class='metric-card'><h2 style='color: black;'>{confidence:.1%}</h2><p style='color: black;'>Confidence</p></div>", 
                   unsafe_allow_html=True)

    # Confidence visualization with native Streamlit
    st.markdown("**Confidence Analysis:**")
    confidence_data = pd.DataFrame({
        'Severity': ['Minor', 'Moderate', 'Severe'],
        'Probability': prediction[0]
    })
    
    # Create a simple bar chart using Streamlit's native chart
    st.bar_chart(confidence_data.set_index('Severity'))
    
    st.markdown("---")
       
    st.markdown('</div>', unsafe_allow_html=True)

# --- Definisi Halaman ---

def cover_page():
    """Halaman cover dengan call to action menarik."""
    st.markdown("""
    <div class="hero-header">
        <h1 >🚗 Car Grader</h1>
        <p>Deteksi Keparahan Kerusakan Mobil</p>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Deteksi kerusakan kendaraan dalam hitungan detik dengan teknologi AI terdepan
        </p>
    </div>
    """, unsafe_allow_html=True)
        
    # Call to Action
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h2 style="color: white; margin-bottom: 2rem;">Mulai Assessment Sekarang!</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Damage Assessment", key="cta_main", use_container_width=True):
            st.session_state.page = "Damage Assessment (Upload)"
            st.rerun()
    
        
def damage_assessment_upload_page():
    """Halaman untuk upload file dengan desain modern."""
    st.markdown(f"""
<style>
.modern-card {{
    /* Tambahkan gaya lain untuk kartu jika diperlukan */
    padding: 20px;
    border-radius: 10px;
    background-color: #f0f2f6; /* Contoh warna latar belakang kartu */
}}

.gradient-text {{
    background: linear-gradient(to right, #00008B, #87CEEB); /* Gradasi dari biru tua ke biru muda */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent; /* Untuk browser lain */
    display: inline-block; /* Penting agar gradasi terlihat */
}}
</style>
<div class="modern-card">
    <h2 class="gradient-text">Upload Photo for Damage Assessment</h2>
    <p style='color: black;'>Upload foto kerusakan kendaraan Anda yang jelas untuk mendapatkan assessment yang akurat</p>
</div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Pastikan foto menunjukkan area kerusakan dengan jelas dan pencahayaan yang baik"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.markdown(f"""
            **Image Details:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Format: {uploaded_file.type}
            - File size: {len(uploaded_file.getvalue())/1024:.1f} KB
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.spinner('🤖 AI sedang menganalisis gambar... Mohon tunggu.'):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                prediction = make_prediction(image)
                display_results(image, prediction)

def damage_assessment_camera_page():
    """Halaman untuk kamera dengan desain modern."""
    st.markdown("""
<style>
.modern-card {
    /* Tambahkan gaya lain untuk kartu jika diperlukan */
    padding: 20px;
    border-radius: 10px;
    background-color: #f0f2f6; /* Contoh warna latar belakang kartu */
}

.gradient-text {
    background: linear-gradient(to right, #00008B, #87CEEB); /* Gradasi dari biru tua ke biru muda */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent; /* Untuk browser lain */
    display: inline-block; /* Penting agar gradasi terlihat */
}
</style>
<div class="modern-card">
    <h2 class="gradient-text">Live Camera Assessment</h2>
    <p style='color: black;'>Gunakan kamera perangkat Anda untuk mengambil foto kerusakan secara langsung</p>
</div>
""", unsafe_allow_html=True)
    
    st.info("💡 **Tips untuk foto terbaik:**\n- Pastikan pencahayaan cukup\n- Fokus pada area kerusakan\n- Ambil dari jarak yang sesuai\n- Hindari bayangan")
    
    camera_photo = st.camera_input("📸 Ambil foto kerusakan")

    if camera_photo:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            image = Image.open(camera_photo).convert('RGB')
            st.image(image, caption='Captured Image', use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.spinner('🤖 AI sedang menganalisis gambar... Mohon tunggu.'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                processed_image = preprocess_image(image)
                prediction = make_prediction(image)
                display_results(image, prediction)

def premium_and_garage_page():
    """Halaman untuk estimasi premi dan pencari bengkel dengan desain modern."""
    st.markdown("""
    <div class="modern-card">
        <h2>💼 Premium Impact & Approved Garages</h2>
        <p>Analisis dampak klaim terhadap premi dan temukan bengkel rekanan terpercaya</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Premium Impact Calculator")
        
        damage_level = st.selectbox(
            "Tingkat kerusakan dari assessment:",
            options=['Minor Damage', 'Moderate Damage', 'Severe Damage'],
            help="Pilih berdasarkan hasil assessment sebelumnya"
        )
        
        current_premium = st.number_input(
            "Premi tahunan saat ini (Rp)", 
            min_value=1000000, 
            value=5000000,
            step=100000,
            format="%d"
        )
        
        claim_history = st.selectbox(
            "Riwayat klaim 3 tahun terakhir:",
            ["Tidak ada", "1 klaim", "2 klaim", "3+ klaim"]
        )
        
        if st.button("🔍 Calculate Premium Impact", use_container_width=True):
            # Impact calculation logic
            base_multiplier = {
                'Minor Damage': 1.10, 
                'Moderate Damage': 1.25, 
                'Severe Damage': 1.50
            }
            
            history_multiplier = {
                "Tidak ada": 1.0,
                "1 klaim": 1.1,
                "2 klaim": 1.2,
                "3+ klaim": 1.3
            }
            
            total_multiplier = base_multiplier[damage_level] * history_multiplier[claim_history]
            new_premium = current_premium * total_multiplier
            increase = new_premium - current_premium
            
            # Create visualization with native Streamlit charts
            premium_data = pd.DataFrame({
                'Type': ['Current Premium', 'New Premium'],
                'Amount': [current_premium, new_premium]
            })
            
            st.markdown("**Premium Comparison:**")
            st.bar_chart(premium_data.set_index('Type'))
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "💰 Current Premium",
                    f"Rp {current_premium:,.0f}"
                )
            
            with col_b:
                st.metric(
                    "📊 New Premium",
                    f"Rp {new_premium:,.0f}",
                    delta=f"+{increase:,.0f}"
                )
            
            st.warning("⚠️ Ini adalah estimasi. Penyesuaian premi final akan ditentukan setelah assessment lengkap.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Find Approved Garages")
        
        # Location selector
        city = st.selectbox(
            "Pilih kota:",
            ["Surabaya", "Jakarta", "Bandung", "Semarang", "Yogyakarta"]
        )
        
        # Sample garage data
        garages = pd.DataFrame({
            'Name': ['AutoPro Service Center', 'CarGrade Master Repair', 'Surabaya Central Auto', 'Gemilang Motor', 'Fast Fix Garage'],
            'Rating': [4.8, 4.9, 4.7, 4.6, 4.5],
            'Distance': ['2.1 km', '3.5 km', '1.8 km', '4.2 km', '2.9 km'],
            'Speciality': ['Body Repair', 'Engine', 'Paint & Body', 'All Services', 'Quick Service'],
            'Phone': ['(031) 123-4567', '(031) 234-5678', '(031) 345-6789', '(031) 456-7890', '(031) 567-8901'],
            'lat': [-7.2756, -7.2905, -7.2575, -7.3300, -7.2850],
            'lon': [112.7942, 112.7981, 112.7508, 112.7200, 112.7700]
        })
        
        # Map visualization
        st.markdown("**📍 Location Map:**")
        st.map(garages[['lat', 'lon']], zoom=11)
        
        # Garage list
        st.markdown("**🏪 Recommended Garages:**")
        for idx, row in garages.iterrows():
            with st.expander(f"⭐ {row['Name']} - Rating: {row['Rating']}"):
                col_x, col_y = st.columns([1, 1])
                with col_x:
                    st.write(f"📍 **Distance:** {row['Distance']}")
                    st.write(f"🔧 **Speciality:** {row['Speciality']}")
                with col_y:
                    st.write(f"📞 **Phone:** {row['Phone']}")
                    st.button(f"📞 Call {row['Name']}", key=f"call_{idx}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar & Navigation ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🚗 Car Grader</h2>
        <p style="font-size: 0.9rem; opacity: 0.8;">AI Insurance Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Navigation menu
    pages = [
        ("🏠 Home", "Home"),
        ("📤 Upload Assessment", "Damage Assessment (Upload)"),
        ("📷 Camera Assessment", "Damage Assessment (Camera)"),
    ]
    
    st.markdown("### 📋 Navigation")
    for display_name, page_key in pages:
        if st.button(display_name, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.rerun()
    
    st.markdown("---")
    
      
    st.markdown("---")
    
    # Contact info
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h4>📞 Emergency Contact</h4>
        <h3>081234196677</h3>
        <p style="font-size: 0.8rem; opacity: 0.7;">© 2025 Car Grader</p>
    </div>
    """, unsafe_allow_html=True)

# --- Page Routing ---
current_page = st.session_state.get('page', 'Home')

if current_page == "Home":
    cover_page()
elif current_page == "Damage Assessment (Upload)":
    damage_assessment_upload_page()
elif current_page == "Damage Assessment (Camera)":
    damage_assessment_camera_page()

# --- Footer ---
st.markdown("""
<div style="text-align: center; padding: 2rem; color: white; opacity: 0.7;">
    <p>© 2025 Car Grader</p>
</div>
""", unsafe_allow_html=True)
