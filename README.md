# Egg Quality

## Clasificación inteligente de huevos mediante visión por computador

**Proyecto académico desarrollado por José Alejandro Tellez Prada y Santiago Perez.**

Egg Quality es una solución de aprendizaje automático orientada a identificar visualmente si un huevo se encuentra en buen estado (`good`) o presenta indicios de agrietamiento (`crack`). El proyecto combina procesamiento digital de imágenes, redes neuronales densas y redes neuronales convolucionales para construir un flujo completo: desde la preparación del dataset hasta una aplicación móvil capaz de enviar imágenes a un modelo entrenado.

El objetivo no es reemplazar una inspección especializada, sino crear una herramienta de apoyo que permita automatizar una primera revisión de calidad de forma rápida, reproducible y accesible desde un celular.

## Objetivo del proyecto

Desarrollar y evaluar un clasificador de imágenes que:

- Detecte huevos aparentemente buenos y huevos posiblemente agrietados.
- Utilice GPU en Google Colab para acelerar el entrenamiento.
- Compare una red MLP con una CNN y documente sus diferencias.
- Priorice el recall de la clase `crack`, ya que es especialmente importante detectar los huevos agrietados.
- Exponga el modelo mediante una API para conectarlo con una aplicación móvil.

## Fuente del dataset

El proyecto utiliza el dataset público **egg Object Detection Dataset**, disponible en Roboflow Universe:

- **Fuente:** [Roboflow Universe - egg-pisqc](https://universe.roboflow.com/muhammad-fauzan-wbvuk/egg-pisqc)
- **Autor:** Muhammad fauzan
- **Tipo:** detección de objetos
- **Contenido:** 408 imágenes, con las clases `good` y `crack`
- **Versiones disponibles:** 2
- **Licencia indicada por la fuente:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

El notebook descarga las imágenes y anotaciones en formato COCO mediante la API de Roboflow. Después utiliza los bounding boxes del dataset para recortar cada huevo, redimensionarlo y preparar los datos para el entrenamiento. La atribución del dataset se conserva aquí para reconocer la autoría y las condiciones de uso de la fuente original.

## Arquitectura

```text
Dataset Roboflow
	|
	v
Google Colab + TensorFlow + GPU T4
	|
	v
Modelo Keras (.keras)
	|
	v
API FastAPI <----- Aplicación Expo Go en el celular
```

## Componentes del repositorio

- `clasificador_huevos_colab.ipynb`: notebook educativo para descarga, preprocesamiento, entrenamiento (MLP vs CNN), evaluación y exportación del modelo. **Flujo completo reconstruido (21 celdas)**.
- `api/`: servidor FastAPI que carga el modelo y recibe imágenes para clasificación.
- `mobile/`: aplicación Expo/React Native con cámara, galería, vista previa y resultado.
- `docs/`: documentación técnica — visión general, arquitectura, dataset, preprocesamiento (incluye filtro **Sobel** creativo), modelos y deploy en VM.
- `handoff.md`: estado del proyecto, decisiones técnicas y tareas pendientes.
- `.gitignore`: exclusiones para evitar subir secretos, datasets, modelos pesados y dependencias generadas.

## Creatividad destacada (sustentación)

1. **Filtro Sobel manual** (`cv2.filter2D`) aplicado antes de la CNN para resaltar grietas — puente entre procesamiento clásico de imágenes y deep learning.
2. **Comparativa justa MLP vs CNN** con los mismos datos, aumentación y callbacks.
3. **Recall de `crack` como criterio de negocio**: un huevo agrietado que pasa por bueno es el error más caro.
4. **Predicción visual en test** (10 ejemplos con confianza) + matrices de confusión por clase.
5. **Contrato de preprocesamiento documentado** y replicado idénticamente en la API.
6. **GPU T4 explícita** (`nvidia-smi`) y tiempo de entrenamiento medido por modelo.

Detalles en `docs/05-creatividad.md`.

## Estado actual

El esqueleto del MVP fue fortalecido: el notebook quedó con flujo completo (21 celdas, listo para ejecutar en Colab), la documentación técnica (`docs/`) cubre arquitectura, dataset, preprocesamiento y creatividad, y la API + app móvil están preparadas para comunicarse. El siguiente hito es ejecutar el entrenamiento en Colab, obtener el modelo final y validar el flujo completo desde el celular.

## Conectar la app al celular

1. Ejecuta el notebook en Google Colab y copia `mejor_cnn.keras` dentro de `api/`.
2. Instala las dependencias de `api/` y levanta FastAPI en la IP local de tu computador:

```powershell
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Busca la IPv4 de tu computador con `ipconfig`.
4. En `mobile/App.js`, cambia `API_URL` por `http://TU_IP_LOCAL:8000`.
5. Con el celular y el computador en la misma red Wi-Fi, ejecuta:

```powershell
cd mobile
npm install
npx expo start
```

6. Abre el proyecto con Expo Go y prueba cámara o galería.

El endpoint `GET /health` sirve para comprobar que la API está visible y que el modelo fue cargado. El endpoint `POST /predict` recibe el archivo de imagen en el campo `file`.

## Flujo de trabajo

1. Roboflow proporciona las imágenes y anotaciones COCO.
2. El notebook recorta los huevos mediante bounding boxes y los normaliza a `100x100` píxeles.
3. TensorFlow entrena una MLP y una CNN usando la GPU disponible en Colab.
4. Se comparan accuracy, recall de `crack`, parámetros y tiempo de entrenamiento.
5. El mejor modelo se exporta para inferencia.
6. La app móvil captura o selecciona una imagen y la envía a FastAPI.
7. La API devuelve la clase estimada y sus probabilidades.

## Alcance y limitaciones actuales

El modelo fue diseñado con imágenes recortadas por bounding box. Por eso, en la primera versión la foto debe mostrar un huevo bien encuadrado. La detección automática del huevo antes de clasificarlo queda como una mejora posterior.

La predicción es una estimación basada en imágenes y no constituye un diagnóstico definitivo de calidad o seguridad alimentaria.

## Autores

- **José Alejandro Tellez Prada**
- **Santiago Perez Florez**
