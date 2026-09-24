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

- `clasificador_huevos_colab.ipynb`: notebook educativo para descarga, preprocesamiento, entrenamiento (MLP vs CNN vs MobileNetV2 con transfer learning), evaluación y exportación del modelo. **Flujo completo (23 celdas)**, con `pip install` al inicio y listo para Colab con GPU.
- `scripts/reentrenar_huevos_colab.py`: script para Colab que **combina varios datasets de Roboflow** y entrena con **fotos completas** (no crops), para que el modelo coincida con lo que envía la app. Exporta el mejor modelo (CNN o MobileNetV2) por recall de `crack`.
- `api/`: servidor FastAPI que carga el modelo y recibe imágenes para clasificación.
- `mobile/`: aplicación Expo/React Native con cámara en vivo (preview continuo + captura), foto, galería, vista previa y resultado.
- `docs/`: documentación técnica — visión general, arquitectura, dataset, preprocesamiento (incluye filtro **Sobel** creativo), modelos (MLP/CNN/**transfer learning**) y deploy en **AWS EC2 con IP elástica**.
- `handoff.md`: estado del proyecto, decisiones técnicas y tareas pendientes.
- `.gitignore`: exclusiones para evitar subir secretos, datasets, modelos pesados y dependencias generadas.

## Creatividad destacada (sustentación)

1. **Filtro Sobel manual** (`cv2.filter2D`) aplicado antes de la CNN para resaltar grietas — puente entre procesamiento clásico de imágenes y deep learning.
2. **Comparativa justa MLP vs CNN** con los mismos datos, aumentación y callbacks.
3. **Recall de `crack` como criterio de negocio**: un huevo agrietado que pasa por bueno es el error más caro.
4. **Predicción visual en test** (10 ejemplos con confianza) + matrices de confusión por clase.
5. **Contrato de preprocesamiento documentado** y replicado idénticamente en la API.
5. **Contrato de preprocesamiento documentado** y replicado idénticamente en la API.
6. **GPU T4 explícita** (`nvidia-smi`) y tiempo de entrenamiento medido por modelo.
7. **Transfer learning con MobileNetV2** como tercer modelo: pesos de ImageNet congelados + cabeza densa entrenada; supera a una CNN desde cero con pocos datos.
8. **Augmentation solo en train** (rotación, flip, zoom) — variaciones de un dataset pequeño sin contaminar la evaluación.
9. **Cámara en vivo** en la app: preview de video continuo con captura manual del fotograma, además de foto y galería.
10. **Deploy en AWS**: EC2 + IP elástica + puerto 8000 abierto + systemd.

Detalles en `docs/05-creatividad.md`.

## Estado actual

El esqueleto del MVP fue fortalecido: el notebook quedó con flujo completo (21 celdas, listo para ejecutar en Colab), la documentación técnica (`docs/`) cubre arquitectura, dataset, preprocesamiento y creatividad, y la API + app móvil están preparadas para comunicarse. El siguiente hito es ejecutar el entrenamiento en Colab, obtener el modelo final y validar el flujo completo desde el celular.

## Conectar la app al celular (vía AWS)

La ruta de deploy completa está en `docs/06-deploy-vm.md`. La EC2 ya existe y su **IP elástica es `54.227.194.211`**. Resumen:

1. Ejecuta el notebook en Colab (GPU T4): entrena MLP, CNN y MobileNetV2, elige el mejor por recall de `crack` y guarda `egg_quality_models.zip` en Drive.
2. Descarga el zip de Drive y cópialo a la VM EC2 en AWS (`scp`).
3. En la EC2: instala `api/requirements.txt`, define `MODEL_PATH` y deja el servicio activo:

```bash
MODEL_PATH=/home/ubuntu/egg_quality_models/mejor_cnn.keras \
uvicorn main:app --host 0.0.0.0 --port 8000
```

O en un solo comando desde tu PC (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_local.ps1 -Pem C:\ruta\llave.pem -Zip .\egg_quality_models.zip -ModelFile mejor_cnn.keras
```

4. Security group: abre el **puerto 8000** (Custom TCP → 0.0.0.0/0) — ya está abierto.
5. Asocia una **IP elástica** a la instancia (la IP no cambia al apagar/encender) — ya está asociada: `54.227.194.211`.
6. En la app (botón ⚙️ Configurar API) usa `http://54.227.194.211:8000`.

Con eso la app funciona desde cualquier red: la IP elástica es pública y el puerto está abierto. Verifica primero `http://54.227.194.211:8000/health` desde el navegador del celular.

> **Nota del modelo**: los números reales del notebook favorecen a **MobileNetV2** (92% accuracy, recall crack 89%, balanceado) sobre la CNN (64% accuracy, predice crack casi siempre). Si la demo se siente "todo crack", reexporta MobileNetV2 con el criterio corregido de la celda 22 y despliega con `-ModelFile mejor_mobilenet.keras`.

## Flujo de trabajo

1. Roboflow proporciona las imágenes y anotaciones COCO.
2. El notebook recorta los huevos mediante bounding boxes y los normaliza a `100x100` píxeles.
3. TensorFlow entrena una MLP, una CNN y una MobileNetV2 con transfer learning usando la GPU disponible en Colab.
4. Se comparan accuracy, recall de `crack`, parámetros y tiempo de entrenamiento.
5. El mejor modelo se exporta a Drive y se despliega en una VM de AWS con IP elástica.
6. La app móvil captura el fotograma de la cámara en vivo, toma una foto o selecciona una imagen, y la envía a FastAPI.
7. La API devuelve la clase estimada y sus probabilidades.

## Alcance y limitaciones actuales

La primera versión entrenaba con imágenes recortadas por bounding box (crops), mientras la app enviaba la foto completa; ese desajuste degradaba la precisión con huevos reales. El script `scripts/reentrenar_huevos_colab.py` corrige esto entrenando con **fotos completas** (como las que manda la app) y ampliando el dataset con más fuentes de Roboflow.

Aun así, la predicción es una estimación basada en imágenes y no constituye un diagnóstico definitivo de calidad o seguridad alimentaria.

## Autores

- **José Alejandro Tellez Prada**
- **Santiago Perez Florez**
