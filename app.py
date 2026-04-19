import streamlit as st
import numpy as np
from PIL import Image

# --- 1. PAGE CONFIGURATION (Loads instantly) ---
st.set_page_config(page_title="AI CropGuard | Smart Agriculture", page_icon="🌿", layout="centered")

# --- 2. THE 38 CLASS DICTIONARY ---
CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy', 
    'Corn - Cercospora Leaf Spot / Gray Leaf Spot', 'Corn - Common Rust', 
    'Corn - Northern Leaf Blight', 'Corn - Healthy', 'Grape - Black Rot', 
    'Grape - Esca (Black Measles)', 'Grape - Leaf Blight', 'Grape - Healthy', 
    'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy', 
    'Pepper Bell - Bacterial Spot', 'Pepper Bell - Healthy', 'Potato - Early Blight', 
    'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy', 
    'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy', 
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 
    'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 
    'Tomato - Healthy'
]

# --- 3. LAZY LOAD THE AI MODEL ---
# This ONLY runs when needed, and saves it in memory so it doesn't freeze twice
@st.cache_resource
def get_ai_model():
    import tensorflow as tf
    return tf.keras.models.load_model('plant_model_38_FINAL_V2.h5')

# --- 4. INSTANT UI RENDER ---
st.title("🌿 AI-Driven Smart Agriculture Platform")
st.markdown("""
Welcome to **CropGuard**. Upload an image of a plant leaf to instantly identify its health status across 38 distinct crop categories. 
*Our preprocessing engine automatically removes real-world backgrounds for higher accuracy.*
""")
st.divider()

# --- 5. THE APPLICATION LOGIC ---
uploaded_file = st.file_uploader("Upload a Leaf Image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    # Updated to width='stretch'
    st.image(original_image, caption="Original Uploaded Image", width='stretch')
    
    # Updated to width='stretch'
    if st.button("🔍 Analyze Leaf Cellular Structure", width='stretch'):
        
        with st.spinner("Initializing AI tools & removing background (First run takes a minute)..."):
            
            from rembg import remove
            
            # 1. Background Removal
            image_no_bg = remove(original_image)
            
            # 2. Apply White Canvas
            white_canvas = Image.new('RGB', image_no_bg.size, (255, 255, 255))
            if image_no_bg.mode == 'RGBA':
                white_canvas.paste(image_no_bg, mask=image_no_bg.split()[3])
            else:
                white_canvas.paste(image_no_bg)
            
            # Updated to width='stretch'
            st.image(white_canvas, caption="Preprocessed Image (AI Vision Layer)", width='stretch')
            
            # 3. Format math for the AI
            final_image = white_canvas.resize((256, 256))
            image_array = np.array(final_image)
            image_array = np.expand_dims(image_array, axis=0) / 255.0
            
            # 4. Load AI and Predict
            model = get_ai_model()
            predictions = model.predict(image_array)
            predicted_class_index = np.argmax(predictions)
            confidence_score = np.max(predictions) * 100
            
            st.divider()
            
            # 5. Output Results
            if confidence_score < 65.0:
                st.error("⚠️ **Diagnosis Unclear**")
                st.write(f"**Confidence Score:** {confidence_score:.2f}%")
                st.info("The AI is not confident enough to make a safe diagnosis. Please upload a clearer photo.")
            else:
                predicted_disease = CLASS_NAMES[predicted_class_index]
                if "Healthy" in predicted_disease:
                    st.success(f"🌱 **Diagnosis:** {predicted_disease}")
                else:
                    st.error(f"🦠 **Diagnosis Detected:** {predicted_disease}")
                st.write(f"📊 **AI Confidence Score:** {confidence_score:.2f}%")