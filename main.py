from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import io

# Load the saved model
model_path = "mad_classifier_model.h5"
model = load_model(model_path)

# FastAPI app
app = FastAPI()

# Preprocess the image
def preprocess_image(file):
    # Read the file as bytes
    image_bytes = file.read()

    # Load the image
    img = load_img(io.BytesIO(image_bytes), target_size=(224, 224)) 
    img = img_to_array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

# Endpoint to classify an uploaded image
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    # Preprocess the image
    img = preprocess_image(file.file)
    
    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
    
    # class labels mapping
    class_labels = {
    0: '0.1 MAD',
    1: '0.2 MAD',
    2: '0.5 MAD',
    3: '1 MAD',
    4: '10 MAD',
    5: '100 MAD',
    6: '2 MAD',
    7: '20 MAD',
    8: '200 MAD',
    9: '5 MAD',
    10: '50 MAD'
}


    # Get the predicted class label
    predicted_label = class_labels.get(predicted_class, "Unknown")

    # Get the model's accuracy
    accuracy = float(predictions[0][predicted_class])

    return {"predicted_class": predicted_label, "accuracy": accuracy}
