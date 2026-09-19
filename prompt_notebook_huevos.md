# Prompt: Notebook Colab (GPU) — Clasificador de huevos good/crack

Copia TODO el bloque de abajo y pégalo en cualquier IA (ChatGPT, Claude, Gemini, Copilot)
para que genere el cuaderno `.ipynb` listo para ejecutar en Google Colab con GPU (T4).

ℹ️ Este archivo (y el notebook que genere la IA) se usará en OTRA carpeta de esta PC / Google Colab,
NO va a correr dentro de este repositorio. Este repositorio solo sirve como **referencia de estilo y nivel**.

---

```
Actúa como un profesor asistente de Ciencia de Datos (estilo UNAB, material del prof. Alfredo Díaz)
que genera cuadernos Jupyter (.ipynb) educativos en español, con celdas Markdown teóricas y celdas
de código comentadas. El cuaderno debe correr de principio a fin en Google Colab con GPU (T4),
sin errores, solo pidiendo al usuario la API key de Roboflow y montar su Drive.

## CONTEXTO Y FUENTE DE ESTILO (importante)
- El notebook generado se ejecutará en Google Colab (GPU T4) en OTRA carpeta de la PC del alumno:
  NO va a correr dentro de la ruta de referencia; NO copies ni importes código desde ahí, solo úsala
  como guía de estilo, nivel y comentarios.
- Ruta de referencia de los apuntes de clase del alumno (para imitar formato y nivel):
  D:\José Tellez\Documents\universidad\sexto semestre\ciencia de datos\Apuntes-ciencia-de-datos-
- Archivos concretos a consultar para replicar el estilo:
  * Redes densas/Avanzada_Cuaderno_4_ANN_Red_Neuronal_Clasificación_(Redes_densas).ipynb
    (teoría: activaciones, funciones de pérdida, batch, early stopping).
  * reconocimiento_de_prendas.ipynb (Fashion MNIST: Flatten -> Dense relu -> softmax, normalizar /255,
    guardado en Google Drive).
  * procesamiento-de-imagenes/Avanzada Cuaderno 6 CNN Procesamiento digital de imágenes.ipynb
    (cv2.resize, reducción de resolución, filtros/convoluciones).
  * procesamiento-de-imagenes/Avanzada_Cuaderno_7_CNN_Redes_Neuronales_Convolucionales_.ipynb
    (Conv2D + MaxPooling2D + Flatten, clasificación CIFAR-10).

## OBJETIVO
Clasificar imágenes de huevos en 2 clases: "good" (buenos) vs "crack" (agrietados), usando el
dataset Roboflow "egg-pisqc" (workspace: muhammad-fauzan-wbvuk, 408 imágenes, 2 clases).
El alumno debe demostrar 3 cosas aprendidas en clase:
(1) preprocesamiento digital de imágenes (redimensionar, recortar, reducir resolución, normalizar),
(2) redes densas (MLP) tipo Fashion MNIST, y (3) redes convolucionales (CNN) tipo CIFAR-10.
El prof. valora creatividad, uso de GPU de Colab y documentación del PORQUÉ de cada decisión
(no solo el qué). El entrenamiento puede durar 2-4 horas, pero justificar cómo lograr el mejor
desempeño en el menor tiempo posible.

## PARTE 0 — CONFIGURACIÓN
1. Título y descripción en Markdown.
2. Verificar GPU: `!nvidia-smi` y `tf.config.list_physical_devices('GPU')`.
3. `!pip install roboflow` si hace falta.
4. API KEY de Roboflow: UNA celda marcada "PEGA AQUÍ TU API KEY" que use
   `from getpass import getpass; API_KEY = getpass("API key de Roboflow: ")` o variable
   directa `API_KEY = "..."`. Explicar en Markdown que la key se obtiene en
   https://app.roboflow.com/settings/api (Settings → API Keys).
5. Montar Drive: `drive.mount('/content/drive')` (ahí se guardará el mejor modelo).

## PARTE 1 — DESCARGA DEL DATASET
- `from roboflow import Roboflow; rf = Roboflow(api_key=API_KEY)`
- `project = rf.workspace("muhammad-fauzan-wbvuk").project("egg-pisqc")`
- Mostrar `project.versions()`, usar versión 1 (fallback versión 2 si no existe).
- Descargar en formato `"coco"` (es detección de objetos con bounding boxes y queremos
  preprocesar nosotros mismos, como en clase).
- Imprimir la estructura de carpetas resultante.

## PARTE 2 — PREPROCESAMIENTO (lo del Cuaderno 6, con visualizaciones)
1. Cargar `_annotations.coco.json` con `json` y explicar su estructura.
2. Función `recortar_por_bbox(...)`: leer con `cv2`, recortar cada bounding box con margen
   2-5%, convertir BGR→RGB, redimensionar a 100x100 con `cv2.resize`, guardar recortes en
   `crops/{train,valid,test}/{good,crack}/`. Explicar POR QUÉ (costo computacional,
   estandarización, menos sobreajuste).
3. Visualización ANTES vs DESPUÉS: original con bbox dibujado (`cv2.rectangle`), recorte
   redimensionado, versión normalizada /255, mosaico 5x5 de ejemplos por clase (estilo
   Fashion MNIST).
4. Reportar balance de clases por split y proponer estrategia si está desbalanceado
   (class_weight o augmentation dirigida).

## PARTE 3 — CARGA DE DATOS
- `tf.keras.utils.image_dataset_from_directory` sobre los crops.
- Normalizar /255 como en Fashion MNIST.
- `.batch(32).prefetch(tf.data.AUTOTUNE)` explicando cada cosa.
- Data augmentation (creatividad): `RandomFlip`, `RandomRotation`, `RandomZoom`, justificando
  por qué ayuda con solo 408 imágenes.

## PARTE 4 — MODELO 1: RED DENSA (MLP)
- `Input((100,100,3)) → Flatten → Dense(128, relu) → Dense(64, relu) → Dense(32, relu)
  → Dense(2, softmax)` (o 1 neurona sigmoid, justificando).
- Markdown: por qué Flatten, por qué ReLU oculto y softmax salida (Cuaderno 4).
- Compilar: `adam` + `sparse_categorical_crossentropy` + `accuracy` (explicar pérdida con
  etiquetas enteras).
- Callbacks explicados: `EarlyStopping(monitor='val_loss', patience=5,
  restore_best_weights=True)`, `ReduceLROnPlateau`, `ModelCheckpoint` → solo mejor modelo
  en `/content/mejor_mlp.keras`.
- Entrenar `epochs=50`, `validation_data`, `batch_size=32`, medir tiempo con `time`.

## PARTE 5 — MODELO 2: CNN
- Estilo Cuaderno 7 (CIFAR-10):
  `Conv2D(32,(3,3),relu) → MaxPooling2D((2,2)) → Conv2D(64,(3,3),relu) → MaxPooling2D((2,2))
  → Conv2D(64,(3,3),relu) → Flatten → Dense(64,relu) → Dropout(0.5) → Dense(2,softmax)`.
- Markdown: teoría CNN (filtros/bordes/patrones locales, pooling, por qué superan a densas en
  imágenes). Justificar dropout contra overfitting.
- Mismos callbacks, mejor modelo en `/content/mejor_cnn.keras`.

## PARTE 6 — COMPARACIÓN Y EVALUACIÓN
1. Graficar curvas loss/accuracy + val de ambos modelos lado a lado; detectar overfitting y
   explicar las medidas tomadas (early stopping, dropout, augmentation).
2. `.evaluate()` en test para ambos.
3. `classification_report` + matriz de confusión para cada uno. Explicar que el recall de la
   clase "crack" importa más (un huevo agrietado que pasa por bueno es un error caro).
4. Predicción visual: 10 imágenes de test con predicción y probabilidad (verde "good ✓" /
   rojo "crack ✗").
5. Tabla comparativa final (accuracy, recall crack, params, tiempo) JUSTIFICANDO cuál modelo
   gana y por qué.

## PARTE 7 — PERSISTENCIA / APP / VM
1. Copiar el mejor modelo a `/content/drive/MyDrive/egg_quality/` con `shutil.copy`, verificar
   con `os.path.exists`. Guardar también `clases.json` ("good","crack") y `preprocesamiento.json`
   (tamaño 100x100, /255, formato color).
2. Comprimir todo en `egg_quality_models.zip`.
3. Markdown final "CÓMO SUBIR A LA VM": instrucciones para descargar el zip desde Drive en la VM
   (`gdown <id>` o descarga manual), descomprimir, instalar tensorflow y
   `tf.keras.models.load_model`.
4. Documentar el CONTRATO DE PREPROCESAMIENTO para la futura app: la app debe aplicar el MISMO
   preprocesamiento (recortar → 100x100 → /255) ANTES de enviar la imagen al servidor (mandar
   algo optimizado, como se aprendió en clase).

## REGLAS
- Español, Markdown teórico conciso con fórmulas, código comentado línea por línea.
- `np.random.seed(42)` y `tf.random.set_seed(42)`.
- Gráficas con título y `plt.show()`; suprimir o documentar warnings.
- Compatible con TensorFlow 2.x de Colab (T4) sin instalar nada raro.
- Entregar SOLO el .ipynb completo, sección por sección, sin omitir ninguna.
```

---

## Notas rápidas

- **API key de Roboflow:** se obtiene en https://app.roboflow.com/settings/api (Settings → API Keys → Copiar, empieza con `rf_`). En el notebook va en la celda de la **Parte 0, punto 4** (una sola celda, con `getpass` para que no quede guardada en el archivo).
- **Subir el modelo a la VM después:** el notebook guarda `egg_quality_models.zip` en Google Drive (`MyDrive/egg_quality/`). Cuando te conectes a la VM, descarga el zip desde Drive (con `gdown <id>` o descarga manual), descomprime, instala tensorflow y carga con `tf.keras.models.load_model("mejor_cnn.keras")`.
- **¿Dónde corre el notebook?** En Google Colab, en OTRA carpeta de la PC, no dentro de este repositorio.
  Este `.md` es solo la guía; el repositorio de apuntes (`D:\José Tellez\...\Apuntes-ciencia-de-datos-`)
  sirve como referencia de estilo para la IA (lee los cuadernos de Fashion MNIST, redes densas y CNN
  para imitar comentarios y nivel).
- **Tip de creatividad para el profe:** pídele a la IA que en la Parte 2 demuestre también un filtro de convolución manual con `cv2.filter2D` (blur o Sobel para resaltar grietas) antes de la CNN — es exactamente el contenido del Cuaderno 6 y muestra relación entre preprocesamiento clásico y deep learning.