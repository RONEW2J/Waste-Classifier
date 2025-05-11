import numpy as np
from PIL import Image
import tensorflow as tf

def predict_image(image_path):
    model = tf.keras.models.load_model('models/waste_classifier.h5')
    img = Image.open(image_path).resize((224, 224))
    img_array = np.expand_dims(np.array(img), 0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    pred = model.predict(img_array)
    return np.argmax(pred)

# Пример использования
print(predict_image('test_image.jpg'))