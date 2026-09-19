import io
import os
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Variables de entorno con valores por defecto
MODEL_PATH = Path(os.getenv('MODEL_PATH', 'mejor_cnn.keras'))
MAX_FILE_MB = int(os.getenv('MAX_FILE_MB', '5'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0')
ALLOW_ORIGINS = os.getenv('ALLOW_ORIGINS')

CLASS_NAMES = ['good', 'crack']
IMAGE_SIZE = (100, 100)

# Configuración CORS: usar ALLOW_ORIGINS si está definido, sino '*'
if ALLOW_ORIGINS:
    origins_list = [origin.strip() for origin in ALLOW_ORIGINS.split(',')]
else:
    origins_list = ['*']

app = FastAPI(title='Egg Quality API', version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

model = None
if MODEL_PATH.exists():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f'Modelo cargado exitosamente: {MODEL_PATH}')
    except Exception as e:
        logger.error(f'Error cargando modelo: {e}')
        model = None


@app.get('/health')
def health():
    return {
        'status': 'ok' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_path': str(MODEL_PATH),
        'version': MODEL_VERSION
    }


@app.get('/model-info')
def model_info():
    """Información del modelo cargado y requisitos de preprocesamiento."""
    if model is None:
        raise HTTPException(status_code=503, detail='Modelo no cargado')
    return {
        'model_name': MODEL_VERSION,
        'input_shape': [*IMAGE_SIZE, 3],
        'classes': CLASS_NAMES,
        'format': 'RGB 100x100 /255',
        'model_path': str(MODEL_PATH),
        'version': MODEL_VERSION
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail='Modelo no cargado. Copia mejor_cnn.keras junto a main.py o define MODEL_PATH.')

    # Validación de tamaño de archivo
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f'Archivo demasiado grande: {file_size_mb:.2f} MB. Máximo permitido: {MAX_FILE_MB} MB.'
        )

    # Validación de tipo de contenido
    if file.content_type not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(status_code=415, detail='Sube una imagen JPG, PNG o WEBP.')

    try:
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
        logger.error(f'Error procesando imagen: {error}')
        raise HTTPException(status_code=400, detail=f'No se pudo procesar la imagen: {error}') from error
