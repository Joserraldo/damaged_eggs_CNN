# =============================================================================
# Egg Quality - Reentrenar con FOTOS COMPLETAS + datasets adicionales de Roboflow
# -----------------------------------------------------------------------------
# Ejecutar en Google Colab con GPU. Pide la API key de Roboflow (no se guarda).
#
# Diferencia con el notebook original:
#   * El notebook entrenaba con CROPS (recortes del huevo vía bbox). La app móvil
#     manda la FOTO COMPLETA, por eso el modelo fallaba con huevos rotos reales.
#   * Este script entrena con la IMAGEN COMPLETA para coincidir con la app.
#   * Descarga VARIOS datasets de Roboflow para ampliar datos (el original era
#     muy pequeño: 1238 crops de 408 imágenes).
#
# Regla de etiquetado por imagen completa: si TODOS los huevos anotados en una
# imagen comparten la misma clase (todas good o todas crack), la imagen completa
# se etiqueta con esa clase. Si hay mezcla o no hay anotaciones, se descarta.
#
# Al final guarda mejor_mobilenet.keras y mejor_cnn.keras en Drive y te los
# puedes descargar para subirlos a la VM de AWS.
# =============================================================================

# %% [markdown]
# ## 0. Configuración base (GPU + paquetes)

# %%
import os
import json
import time
import shutil
import random
import warnings
from pathlib import Path
from getpass import getpass
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from google.colab import drive
from sklearn.metrics import classification_report, recall_score

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 30
CLASS_NAMES = ['good', 'crack']

print('TensorFlow:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))

# %%
# PEGA AQUÍ TU API KEY (no queda escrita en el archivo).
# Se obtiene en: https://app.roboflow.com/settings/api
API_KEY = getpass('API key de Roboflow: ')
if not API_KEY.strip():
    raise ValueError('La API key no puede estar vacía.')

drive.mount('/content/drive')
DRIVE_DIR = Path('/content/drive/MyDrive/egg_quality')
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Datasets de Roboflow a combinar
#
# Cada entrada: (workspace, project, versión, mapa de clases -> good/crack)
# El mapa recibe el nombre de clase en minúsculas y devuelve 'good' o 'crack'.
# Se usa * (comodín) en las reglas para capturar variantes.

DATASETS = [
    # El dataset original del notebook (referencia)
    ('muhammad-fauzan-wbvuk', 'egg-pisqc', 1, {'crack': 'crack', 'good': 'good'}),
    # Dataset grande con good/demage/dirty
    ('new-workspace-ltyar', 'egg-egg-1hseh', None, {'good': 'good', 'demage': 'crack', 'dirty': None}),
    # normal/damaged
    ('computer-science-7mly8', 'egg-a2ssv', None, {'normal': 'good', 'damaged': 'crack'}),
    # NORMAL/BROKEN/DIRTY
    ('new-workspace-ltyar', 'egg-egg-12', None, {'normal': 'good', 'broken': 'crack', 'dirty': None}),
    # cracked-egg / normal-egg
    ('crackedeggs', 'cracked-eggs', None, {'cracked-egg': 'crack', 'normal-egg': 'good'}),
    # brown/white cracked vs normal
    ('stust-bcxxg', 'egg-dqnud', None, {
        'brown crack': 'crack', 'white crack': 'crack',
        'borken': 'crack',
        'brown': 'good', 'white': 'good',
    }),
    # NORMAL/BROKEN/DIRTY (variante)
    ('new-workspace-ltyar', 'egg-egg-liushuixian', None, {'normal': 'good', 'broken': 'crack', 'dirty': None}),
    # crackeggs / normaleggs / etc
    ('egg-rrh5x', 'egg-4cbmo', None, {
        'crackeggs': 'crack', 'normaleggs': 'good', 'dirtyeggs': None,
        'softshelleg': 'good', 'discoloredeggs': 'good', 'calciumcoatedeggs': 'good',
    }),
]

# %% [markdown]
# ## 2. Descarga de datasets

# %%
from roboflow import Roboflow

rf = Roboflow(api_key=API_KEY)
DOWNLOAD_ROOT = Path('/content/datasets')
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

def normalize_class_name(name):
    """Baja a minúsculas, quita espacios múltiples y subrayados -> espacios."""
    return ' '.join(name.lower().strip().replace('_', ' ').split())

def download_dataset(workspace, project, version_num):
    project_obj = rf.workspace(workspace).project(project)
    versions = list(project_obj.versions())
    if version_num is None:
        # Elegir la versión más reciente que tenga train/valid/test
        version = versions[0]
    else:
        version = next((v for v in versions if getattr(v, 'version', None) == version_num), versions[0])
    dataset = version.download('coco', location=str(DOWNLOAD_ROOT / project))
    return Path(dataset.location)

# Descargamos todos los datasets
dataset_dirs = []
for workspace, project, version_num, _ in DATASETS:
    print(f'\n=== Descargando {workspace}/{project} ===')
    try:
        d = download_dataset(workspace, project, version_num)
        dataset_dirs.append(d)
        print('  ->', d)
    except Exception as e:
        print(f'  !! Error descargando {project}: {e}')
        continue

# %% [markdown]
# ## 3. Construir dataset de FOTOS COMPLETAS
#
# Por cada imagen anotada: si todos los huevos comparten la misma clase
# (good o crack), se copia la IMAGEN COMPLETA a train/valid/test bajo esa clase.

# %%
def find_annotation_file(split_dir):
    files = list(split_dir.glob('_annotations.coco.json'))
    return files[0] if files else None

def find_image(split_dir, file_name):
    candidate = split_dir / file_name
    if candidate.exists():
        return candidate
    matches = list(split_dir.rglob(Path(file_name).name))
    return matches[0] if matches else None

FULL_DIR = Path('/content/full')
FULL_DIR.mkdir(parents=True, exist_ok=True)

# Mapa de clasificación con comodines
def classify_label(raw_name, class_map):
    """Devuelve 'good'/'crack'/None según class_map (con comodín en 'crack'/'good')."""
    name = normalize_class_name(raw_name)
    # Iterar por la clave MÁS ESPECÍFICA primero (más larga) para que
    # 'brown cracked egg' matchee 'brown crack...' antes que el genérico 'brown'.
    for key, value in sorted(class_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        key_norm = normalize_class_name(key)
        if key_norm in name:
            return value
    # Fallback por heurística si no hay coincidencia exacta
    if 'crack' in name or 'broken' in name or 'damag' in name or 'demag' in name:
        return 'crack'
    if 'good' in name or 'normal' in name or 'intact' in name:
        return 'good'
    return None

# Recolecta ejemplos: (ruta_imagen_completa, clase, nombre_imagen_unico)
examples = []
for ds_dir, (_, project, _, class_map) in zip(dataset_dirs, DATASETS):
    for split_name in ['train', 'valid', 'test']:
        split_dir = ds_dir / split_name
        if not split_dir.exists():
            continue
        annotation_path = find_annotation_file(split_dir)
        if annotation_path is None:
            continue
        with annotation_path.open(encoding='utf-8') as f:
            coco = json.load(f)
        categories = {item['id']: item['name'] for item in coco.get('categories', [])}
        images = {item['id']: item for item in coco.get('images', [])}

        # Agrupar anotaciones por imagen
        per_image = defaultdict(list)
        for ann in coco.get('annotations', []):
            per_image[ann['image_id']].append(ann)

        for image_id, anns in per_image.items():
            image_info = images.get(image_id)
            if image_info is None:
                continue
            # Clases de todos los huevos de esta imagen
            labels = set()
            for ann in anns:
                raw = categories.get(ann['category_id'], '')
                cls = classify_label(raw, class_map)
                if cls is not None:
                    labels.add(cls)
            # Solo si TODOS los huevos comparten la misma clase (etiqueta limpia)
            if len(labels) != 1:
                continue
            label = labels.pop()
            img_path = find_image(split_dir, image_info['file_name'])
            if img_path is None:
                continue
            # Nombre único: proyecto_split_imagen
            unique_name = f"{project}_{split_name}_{Path(image_info['file_name']).stem}"
            examples.append((img_path, label, unique_name))

print(f'\nEjemplos (imágenes completas) recolectados: {len(examples)}')
from collections import Counter
print(Counter(label for _, label, _ in examples))

# %% [markdown]
# ## 4. División train/valid/test POR IMAGEN (evita data leakage)

# %%
# Agrupar por nombre de imagen única
groups = defaultdict(list)
for img_path, label, unique_name in examples:
    groups[unique_name].append((img_path, label))

image_groups = list(groups.keys())
random.shuffle(image_groups)
n = len(image_groups)
train_end = int(n * 0.80)
valid_end = int(n * 0.90)
split_groups = {
    'train': image_groups[:train_end],
    'valid': image_groups[train_end:valid_end],
    'test': image_groups[valid_end:],
}

# Copiar imágenes completas a las carpetas por split/clase
for split in ['train', 'valid', 'test']:
    for label in CLASS_NAMES:
        (FULL_DIR / split / label).mkdir(parents=True, exist_ok=True)

for split, group_names in split_groups.items():
    for g in group_names:
        for img_path, label in groups[g]:
            dst = FULL_DIR / split / label / f"{g}.jpg"
            shutil.copy2(img_path, dst)

print('División completada.\n')
for split in ['train', 'valid', 'test']:
    good = len(list((FULL_DIR / split / 'good').glob('*.jpg')))
    crack = len(list((FULL_DIR / split / 'crack').glob('*.jpg')))
    print(f'{split}: good={good} crack={crack}')

# %% [markdown]
# ## 5. Cargar datasets + normalización + augmentación

# %%
def load_split(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        FULL_DIR / split,
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

train_raw = load_split('train', True)
valid_raw = load_split('valid', False)
test_raw = load_split('test', False)

normalization = tf.keras.layers.Rescaling(1.0 / 255)
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
], name='augmentation')
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_train(images, labels):
    return augmentation(normalization(images)), labels

def preprocess_eval(images, labels):
    return normalization(images), labels

train_ds = train_raw.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
valid_ds = valid_raw.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_raw.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# %% [markdown]
# ## 6. Definir modelos (CNN y MobileNetV2, mismas arquitecturas que el notebook)

# %%
def callbacks_for(path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(path, monitor='val_loss', save_best_only=True),
    ]

# CNN desde cero
cnn = tf.keras.Sequential([
    tf.keras.Input(shape=(*IMG_SIZE, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax'),
], name='cnn')
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# MobileNetV2 (transfer learning)
base_mobilenet = tf.keras.applications.MobileNetV2(
    include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
base_mobilenet.trainable = False
mobilenet = tf.keras.Sequential([
    tf.keras.Input(shape=(*IMG_SIZE, 3)),
    tf.keras.layers.Rescaling(2.0, offset=-1.0),
    base_mobilenet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation='softmax'),
], name='mobilenetv2')
mobilenet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %% [markdown]
# ## 7. Entrenar

# %%
print('>>> Entrenando CNN...')
start = time.time()
history_cnn = cnn.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=callbacks_for('/content/mejor_cnn.keras'), verbose=1)
elapsed_cnn = time.time() - start
print(f'CNN entrenada en {elapsed_cnn:.1f} s')

print('>>> Entrenando MobileNetV2...')
start = time.time()
history_mobilenet = mobilenet.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  callbacks=callbacks_for('/content/mejor_mobilenet.keras'), verbose=1)
elapsed_mobilenet = time.time() - start
print(f'MobileNetV2 entrenada en {elapsed_mobilenet:.1f} s')

# %% [markdown]
# ## 8. Evaluar y elegir el mejor modelo (recall_crack >= 0.85)

# %%
def evaluate_model(model, name, elapsed):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    recall_crack = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f'\n=== {name} ===')
    print(f'loss/acc: accuracy={accuracy:.4f} recall_crack={recall_crack:.4f} tiempo={elapsed:.1f}s')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return accuracy, recall_crack

results = {}
acc_cnn, rec_cnn = evaluate_model(cnn, 'CNN', elapsed_cnn)
results['CNN'] = {'accuracy': acc_cnn, 'recall_crack': rec_cnn}
acc_mob, rec_mob = evaluate_model(mobilenet, 'MobileNetV2', elapsed_mobilenet)
results['MobileNetV2'] = {'accuracy': acc_mob, 'recall_crack': rec_mob}

MIN_RECALL_CRACK = 0.85
candidates = {k: r for k, r in results.items() if r['recall_crack'] >= MIN_RECALL_CRACK}
if candidates:
    best_name = max(candidates, key=lambda k: candidates[k]['accuracy'])
else:
    best_name = max(results, key=lambda k: results[k]['recall_crack'])
print(f'\nMejor modelo: {best_name} (accuracy={results[best_name]["accuracy"]:.4f}, recall_crack={results[best_name]["recall_crack"]:.4f})')

best_path = {'CNN': '/content/mejor_cnn.keras', 'MobileNetV2': '/content/mejor_mobilenet.keras'}[best_name]
best_model = {'CNN': cnn, 'MobileNetV2': mobilenet}[best_name]
best_model.save(best_path)

# Guardar en Drive
shutil.copy2(best_path, DRIVE_DIR / Path(best_path).name)
print(f'\nModelo guardado en: {best_path}')
print(f'Copiado a Drive: {DRIVE_DIR / Path(best_path).name}')
print('\nDescarga el .keras de Drive y súbelo a la VM de AWS (o usa deploy_local.ps1).')
