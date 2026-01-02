import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image

# -------- Settings --------
MODEL_PATH = "waste_sorting_model.keras"
IMG_SIZE = (320, 320)
CONF_THRESHOLD = 0.60 
TOP_K = 3

# -------- Load model once (cached) --------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# Class names hardcoded to match the training order
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Category colors for visual feedback
CATEGORY_COLORS = {
    'paper': '#3498db',
    'glass': '#2ecc71',
    'plastic': '#e74c3c',
    'metal': '#95a5a6',
    'trash': '#34495e',
    'cardboard': '#e67e22'
}


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    img: HxWxC in RGB uint8
    returns: (1, H, W, 3) float32 suitable for EfficientNetV2 preprocess
    """
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, axis=0)  # (1,H,W,3)

    # EfficientNetV2 preprocessing (same as training)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return img


def predict(img: np.ndarray):
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)

    # Top-K
    top_idx = np.argsort(probs)[::-1][:TOP_K]
    top = [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]

    # Full probability dict
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    
    return top, prob_dict


# -------- Streamlit UI --------
st.set_page_config(
    page_title="Waste Sorting Assistant",
    page_icon="♻️",
    layout="centered"
)

st.title("Waste Sorting Assistant Demo")
st.markdown(
    "Upload an image of a waste item and the model will predict the category "
    "(cardboard, glass, metal, paper, plastic, trash)."
)

uploaded_file = st.file_uploader(
    "Upload a waste item photo",
    type=["jpg", "jpeg", "png", "webp", "avif"],
    help="Supported formats: JPG, JPEG, PNG, WEBP, AVIF"
)

if uploaded_file is not None:
    # Load and display the image
    from io import BytesIO
    try:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    except Exception as e:
        st.error(
            f"**Could not open image.** The file may be corrupted or in an unsupported format.\n\n"
            f"If this was converted from AVIF, try re-saving it as a proper JPEG/PNG using an image editor.\n\n"
            f"Error: `{e}`"
        )
        st.stop()
    
    img_array = np.array(image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", width="stretch")
    
    with col2:
        with st.spinner("Analyzing..."):
            top_predictions, all_probs = predict(img_array)
        
        best_label, best_conf = top_predictions[0]
        
        # Main prediction result
        st.markdown("### Prediction")
        st.markdown(f"**Category:** `{best_label.upper()}`")
        st.markdown(f"**Confidence:** `{best_conf:.1%}`")
        
        # Low confidence warning
        if best_conf < CONF_THRESHOLD:
            st.warning(
                "**Low confidence** — try a clearer photo "
                "(good lighting, closer object, plain background)."
            )
        
        # Progress bars for top predictions
        st.markdown("### Top Predictions")
        for label, prob in top_predictions:
            st.progress(prob, text=f"{label}: {prob:.1%}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Built with Streamlit & TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
