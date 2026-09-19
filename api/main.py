import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_PATH = Path(os.getenv('MODEL_PATH', 'mejor_cnn.keras'))
CLASS_NAMES = ['good', 'crack']
IMAGE_SIZE = (100, 100)

app = FastAPI(title='Egg Quality API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

model = None
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail='Modelo no encontrado. Copia mejor_cnn.keras junto a main.py o define MODEL_PATH.')
    if file.content_type not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(status_code=415, detail='Sube una imagen JPG, PNG o WEBP.')
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize(IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32) / 255.0
        probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        return {
            'label': CLASS_NAMES[class_index],
            'confidence': float(probabilities[class_index]),
            'probabilities': {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)},
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'No se pudo procesar la imagen: {error}') from error
