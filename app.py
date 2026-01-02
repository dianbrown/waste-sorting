import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Waste Sorting Assistant | CNN Research Project",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# MODEL SETTINGS
# ============================================================================
MODEL_PATH = "waste_sorting_model.keras"
IMG_SIZE = (320, 320)
CONF_THRESHOLD = 0.60
TOP_K = 3
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ============================================================================
# LOAD MODEL (CACHED)
# ============================================================================
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image for EfficientNetV2 inference."""
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return img

def predict(img: np.ndarray):
    """Run inference and return top-K predictions."""
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:TOP_K]
    top = [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return top, prob_dict

# ============================================================================
# HEADER
# ============================================================================
st.markdown("# Waste Sorting Assistant")
st.markdown("##### A CNN-based image classification system for automated waste categorization")
st.markdown("---")

# ============================================================================
# IMAGE CLASSIFICATION SECTION
# ============================================================================
st.markdown("### Image Classification")
st.markdown("Upload an image of a waste item to classify it into one of six categories: **Cardboard**, **Glass**, **Metal**, **Paper**, **Plastic**, or **Trash**.")

uploaded_file = st.file_uploader(
    "Select an image file",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP. Maximum size: 200MB."
)

if uploaded_file is not None:
    from io import BytesIO
    try:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    except Exception as e:
        st.error(
            f"**Could not open image.** The file may be corrupted or in an unsupported format.\n\n"
            f"Error: `{e}`"
        )
        st.stop()
    
    img_array = np.array(image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Input Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown("#### Classification Results")
        
        with st.spinner("Running inference..."):
            top_predictions, all_probs = predict(img_array)
        
        best_label, best_conf = top_predictions[0]
        
        st.markdown(f"**Predicted Category:** `{best_label.upper()}`")
        st.markdown(f"**Confidence Score:** `{best_conf:.2%}`")
        
        if best_conf < CONF_THRESHOLD:
            st.warning(
                "Low confidence prediction. Consider using a clearer image with "
                "better lighting and a plain background."
            )
        
        st.markdown("---")
        st.markdown("##### Probability Distribution")
        
        for label in CLASS_NAMES:
            prob = all_probs[label]
            st.progress(prob, text=f"{label.capitalize()}: {prob:.1%}")

# ============================================================================
# PROJECT OVERVIEW
# ============================================================================
st.markdown("---")
st.markdown("## About This Project")

st.markdown("""
This is my **first Convolutional Neural Network (CNN) project**, developed to explore 
deep learning techniques for image classification in the context of environmental sustainability.

The goal was to build a practical application that demonstrates how AI can assist in 
automating waste sorting, potentially reducing contamination in recycling streams and 
improving overall recycling efficiency.
""")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
st.markdown("---")
st.markdown("### Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Test Accuracy", value="99.88%")

with col2:
    st.metric(label="Validation Loss", value="0.0063")

with col3:
    st.metric(label="Classes", value="6")

# ============================================================================
# SAMPLE DATASET IMAGES
# ============================================================================
st.markdown("---")
st.markdown("### Sample Dataset Images")
st.markdown("Representative samples from each of the six waste categories used for training:")
st.image("assets/sample_dataset_images.png", caption="Sample images from the Garbage Classification dataset", use_container_width=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
st.markdown("---")
st.markdown("### Model Architecture")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Backbone: EfficientNetV2-S")
    st.markdown("""
    **Why EfficientNetV2?**
    
    - **State-of-the-art accuracy/efficiency trade-off**: Achieves high accuracy with fewer parameters and FLOPs compared to alternatives like ResNet or VGG.
    
    - **Optimized for training speed**: Uses progressive learning and improved training techniques built into the architecture.
    
    - **Smaller model size**: More suitable for deployment on resource-constrained environments while maintaining high accuracy.
    
    - **ImageNet pretrained weights**: Leverages knowledge from 1 million+ images for effective transfer learning.
    """)

with col2:
    st.markdown("#### Network Structure")
    st.code("""
Input Layer (320 x 320 x 3)
        |
        v
EfficientNetV2-S (ImageNet pretrained)
        |
        v
Global Average Pooling 2D
        |
        v
Dropout (0.3)
        |
        v
Dense Layer (6 units, Softmax)
        |
        v
Output: Class Probabilities
    """, language=None)

# ============================================================================
# TRAINING METHODOLOGY
# ============================================================================
st.markdown("---")
st.markdown("### Training Methodology")

st.markdown("#### Two-Stage Transfer Learning Approach")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Stage 1: Frozen Backbone Training**
    
    In the first stage, the pretrained EfficientNetV2-S backbone is kept frozen, 
    meaning its weights are not updated during training. Only the newly added 
    classifier head (Dense layers) is trained.
    
    | Parameter | Value |
    |-----------|-------|
    | Learning Rate | 1e-3 |
    | Batch Size | 32 |
    | Epochs | 10 |
    | Optimizer | Adam |
    
    This stage allows the classifier to learn the relationship between the 
    pretrained features and our specific classes without disrupting the 
    learned representations.
    """)

with col2:
    st.markdown("""
    **Stage 2: Fine-Tuning**
    
    After the classifier head converges, the backbone is unfrozen and the 
    entire network is trained end-to-end with a much lower learning rate 
    to prevent catastrophic forgetting.
    
    | Parameter | Value |
    |-----------|-------|
    | Learning Rate | 2e-5 |
    | Batch Size | 8 |
    | Epochs | 30 (early stopping) |
    | BatchNorm Layers | Frozen |
    
    The smaller batch size accommodates the increased memory requirements 
    when computing gradients for all layers.
    """)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
st.markdown("---")
st.markdown("### Model Evaluation")
st.markdown("Confusion matrix showing classification performance across all six categories:")
st.image("assets/confusion_matrix.png", caption="Confusion Matrix - Test Set Results", use_container_width=True)

# ============================================================================
# KEY TECHNIQUES
# ============================================================================
st.markdown("---")
st.markdown("### Key Techniques Employed")

techniques = [
    ("Transfer Learning", "Leveraged ImageNet pretrained weights to reduce training time and improve accuracy with limited data."),
    ("Mixed Precision (FP16)", "Used float16 computations during training for 2x speedup and 50% memory reduction on GPU."),
    ("Data Augmentation", "Applied random flips, rotations, zoom, translation, and contrast adjustments to improve generalization."),
    ("Early Stopping", "Monitored validation accuracy and stopped training when no improvement was observed for 6 epochs."),
    ("Learning Rate Scheduling", "Reduced learning rate by 50% when validation accuracy plateaued for 2 consecutive epochs."),
    ("Dropout Regularization", "Applied 30% dropout before the final layer to prevent overfitting.")
]

for name, description in techniques:
    st.markdown(f"**{name}**")
    st.markdown(f"{description}")
    st.markdown("")

# ============================================================================
# DATASET
# ============================================================================
st.markdown("---")
st.markdown("### Dataset")

st.markdown("""
The model was trained on the [Garbage Classification Dataset](https://www.kaggle.com/datasets/hassnainzaidi/garbage-classification) 
from Kaggle, containing approximately 2,500 images across 6 categories.

The dataset was pre-organized into train, validation, and test splits, eliminating the need for manual partitioning.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    | Category | Description |
    |----------|-------------|
    | Cardboard | Cardboard boxes, packaging materials |
    | Glass | Glass bottles, jars, containers |
    | Metal | Aluminum cans, metal containers |
    """)

with col2:
    st.markdown("""
    | Category | Description |
    |----------|-------------|
    | Paper | Newspapers, documents, magazines |
    | Plastic | Plastic bottles, bags, containers |
    | Trash | Non-recyclable waste items |
    """)

# ============================================================================
# LINKS
# ============================================================================
st.markdown("---")
st.markdown("### Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("[GitHub Repository](https://github.com/dianbrown/waste-sorting)")

with col2:
    st.markdown("[Hugging Face Space](https://huggingface.co/spaces/dianbrown/waste-sorting)")

with col3:
    st.markdown("[Kaggle Dataset](https://www.kaggle.com/datasets/hassnainzaidi/garbage-classification)")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.9rem;'>"
    "Built with TensorFlow and Streamlit | First CNN Project by Dian Brown"
    "</div>",
    unsafe_allow_html=True
)
