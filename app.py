import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# Constants
MODEL_PATH = 'models/waste_classifier.h5'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")


def preprocess_image(img):
    """Consistent preprocessing matching training"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)

    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    return tf.keras.applications.efficientnet.preprocess_input(img_array)


def classify_image(img):
    """Classify waste image with timing"""
    try:
        start_time = time.time()

        # Preprocess
        img_array = preprocess_image(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        pred_time = time.time() - start_time

        # Return only class probabilities (no strings)
        return {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }

    except Exception as e:
        return {"error": str(e)}


# Create interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Waste Image"),
    outputs=gr.Label(num_top_classes=3, label="Classification Results"),
    title="♻️ Waste Classification AI",
    description="Classifies waste into: cardboard, glass, metal, paper, plastic, trash",
    examples=[
        ["examples/cardboard.jpg"],
        ["examples/plastic.jpg"],
        ["examples/glass.jpg"]
    ] if os.path.exists("examples") else None,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )