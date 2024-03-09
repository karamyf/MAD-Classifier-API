# MAD-Classifier-API
This API classifies images of Moroccan currency (MAD) denominations. It uses Convolutional Neural Network (CNN) model to predict the denomination of uploaded images.

## How to Use

You can use this API by sending a POST request with banknotes or coins image. The API will return the predicted denomination and the confidence score (accuracy) of the prediction.

### Endpoint

- URL: [https://sonorus-mad-classifier.hf.space/classify/](https://sonorus-mad-classifier.hf.space/classify/)
- Method: POST
- Headers: `Content-Type: multipart/form-data`
- Body: Form data with key as 'file' and value as the image file

### Example

```python
import requests

url = "https://sonorus-mad-classifier.hf.space/classify/"
files = {"file": open("path_to_your_image.jpg", "rb")}
response = requests.post(url, files=files)
data = response.json()

print("Predicted Denomination:", data["predicted_class"])
print("Confidence Score:", data["accuracy"])
```


### Example Response

```json
{
    "predicted_class": "20 MAD",
    "accuracy": 0.95
}
```

## Installation

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

## Running the API

To run the API locally, use the following command:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000/`.

## Deployment

The API is deployed and accessible at [https://sonorus-mad-classifier.hf.space/classify/](https://sonorus-mad-classifier.hf.space/classify/).

---



**Note:** The `model.h5` file is not included in this repository. You can use your own trained model or use the provided API. Feel free to customize this template to fit your specific needs and add any additional information about your API.
