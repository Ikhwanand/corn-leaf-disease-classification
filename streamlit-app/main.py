import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="🌽 Corn Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .prediction-text {
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Class names and descriptions
CLASS_NAMES = ['Grey Leaf Spot', 'Corn Rust', 'Leaf Blight']
CLASS_DESCRIPTIONS = {
    'Grey Leaf Spot': {
        'description': 'Penyakit yang disebabkan oleh jamur Cercospora zeae-maydis',
        'symptoms': 'Bercak abu-abu persegi panjang pada daun',
        'treatment': 'Fungisida berbahan aktif strobilurin atau triazole'
    },
    'Corn Rust': {
        'description': 'Penyakit yang disebabkan oleh jamur Puccinia sorghi',
        'symptoms': 'Pustula karat berwarna coklat kemerahan pada daun',
        'treatment': 'Fungisida berbahan aktif triazole atau strobilurin'
    },
    'Leaf Blight': {
        'description': 'Penyakit yang disebabkan oleh jamur Exserohilum turcicum',
        'symptoms': 'Bercak coklat memanjang dengan bentuk elips pada daun',
        'treatment': 'Fungisida berbahan aktif mancozeb atau chlorothalonil'
    }
}

@st.cache_resource
def load_onnx_model():
    """Load ONNX model with caching"""
    model_path = os.path.join('../models', 'corn_disease_model.onnx')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    try:
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if ort.get_available_providers():
            # Use GPU if available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL to OpenCV format
    image_array = np.array(image)
    
    # Convert RGB to BGR if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image_array, (224, 224))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def predict_disease(image, session):
    """Predict disease from image using ONNX model"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        predictions = session.run([output_name], {input_name: processed_image})[0]
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        all_probabilities = predictions[0]
        
        return predicted_class_idx, confidence, all_probabilities
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None

def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">🌽 Corn Disease Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Information")
        st.info("""
        **How to use:**
        1. Upload an image of corn leaf
        2. Wait for the analysis
        3. View the prediction results
        
        **Supported formats:**
        - JPG, JPEG, PNG
        - Max size: 200MB
        """)
        
        st.header("🔬 Model Info")
        st.write("**Model Type:** Custom CNN")
        st.write("**Framework:** ONNX Runtime")
        st.write("**Classes:** 3 diseases")
        st.write("**Accuracy:** 98.60%")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image of corn leaf",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of corn leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.write(f"**Image size:** {image.size}")
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.header("🔍 Prediction Results")
        
        if uploaded_file is not None:
            # Load model
            with st.spinner("Loading model..."):
                session = load_onnx_model()
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_idx, confidence, probabilities = predict_disease(image, session)
            
            if predicted_idx is not None:
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence_color = get_confidence_color(confidence)
                
                # Prediction result box
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 class="prediction-text">🎯 Prediction: {predicted_class}</h3>
                    <p class="{confidence_color}">Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                st.subheader("📊 Probability Distribution")
                prob_data = {
                    'Disease': CLASS_NAMES,
                    'Probability': probabilities
                }
                st.bar_chart(prob_data, x='Disease', y='Probability')
                
                # Disease information
                st.subheader("ℹ️ Disease Information")
                disease_info = CLASS_DESCRIPTIONS[predicted_class]
                
                st.write(f"**Description:** {disease_info['description']}")
                st.write(f"**Symptoms:** {disease_info['symptoms']}")
                st.write(f"**Treatment:** {disease_info['treatment']}")
                
                # Confidence interpretation
                st.subheader("🎯 Confidence Interpretation")
                if confidence >= 0.8:
                    st.success("🟢 High confidence - Very reliable prediction")
                elif confidence >= 0.6:
                    st.warning("🟡 Medium confidence - Fairly reliable prediction")
                else:
                    st.error("🔴 Low confidence - Consider retaking the image")
        
        else:
            st.info("👆 Please upload an image to start analysis")
    
    # Additional information
    st.markdown("---")
    st.header("📚 About the Diseases")
    
    # Disease cards
    cols = st.columns(3)
    for i, (disease, info) in enumerate(CLASS_DESCRIPTIONS.items()):
        with cols[i]:
            st.subheader(f"🦠 {disease}")
            st.write(f"**Cause:** {info['description']}")
            st.write(f"**Symptoms:** {info['symptoms']}")
            st.write(f"**Treatment:** {info['treatment']}")

if __name__ == "__main__":
    main()
