# 06 - Deploy en VM

Guía para llevar el modelo entrenado desde Colab hasta la VM donde corre FastAPI.

## 1. En Colab (el notebook lo hace solo)

- Guarda el mejor modelo en `/content/mejor_cnn.keras` (o `mejor_mlp.keras`).
- Exporta a Drive: `MyDrive/egg_quality/models/` + `clases.json` + `preprocesamiento.json`.
- Comprime todo en `egg_quality_models.zip` y lo copia a Drive.

## 2. En la VM

```bash
# Descargar desde Drive (gdown con el id del archivo, o descarga manual)
pip install gdown
gdown <FILE_ID_de_egg_quality_models.zip>

# Descomprimir
unzip egg_quality_models.zip -d egg_quality_models

# Dependencias de la API
cd api
pip install -r requirements.txt      # fastapi, uvicorn, tensorflow, pillow, numpy
```

## 3. Levantar la API

```bash
# MODEL_PATH apunta al modelo descargado
export MODEL_PATH=/ruta/a/egg_quality_models/mejor_cnn.keras
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4. Verificar

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

## 5. Contrato de preprocesamiento (debe coincidir)

| Parámetro | Valor |
|---|---|
| Tamaño | 100x100 |
| Canales | RGB |
| Normalización | /255 |
| Formato API | multipart/form-data, campo `file` (jpg/png/webp) |

Regla crítica: el preprocesamiento de la API debe ser **idéntico** al de entrenamiento. Si la imagen llega a otra resolución o sin normalizar, la predicción no vale. La app móvil muestra instrucciones de encuadre para que el huevo llegue bien centrado (el modelo fue entrenado con crops de bbox).