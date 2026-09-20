# 04 - Modelos

Se entrenan tres arquitecturas con los MISMOS datos, aumentación y callbacks para una comparación justa: MLP (línea base densa), CNN desde cero y MobileNetV2 con transfer learning.

## Callbacks compartidos (PORQUÉ de cada uno)

```python
def callbacks_for(path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(path, monitor='val_loss', save_best_only=True),
    ]
```

| Callback | Función | Por qué aquí |
|---|---|---|
| EarlyStopping | Detiene si val_loss no mejora en 5 épocas | Con 408 imágenes se sobreajusta rápido; evita épocas inútiles |
| ReduceLROnPlateau | Baja LR ×0.3 si val_loss se estanca 2 épocas | Afina el mínimo local sin reiniciar el entrenamiento |
| ModelCheckpoint | Guarda el mejor val_loss | Siempre quedarnos con la mejor versión, no la última |

## MLP (línea base)

```python
mlp = tf.keras.Sequential([
    tf.keras.Input(shape=(100, 100, 3)),
    tf.keras.layers.Flatten(),          # 30.000 valores → vector
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),
], name='mlp')
```

**Por qué**: demuestra redes densas (Cuaderno 4, Fashion MNIST). **Limitación esperada**: no captura estructura espacial; los píxeles se tratan como un vector plano → accuracy y recall inferiores a la CNN. Su valor es la línea base: cuantifica cuánto gana la convolución.

## CNN

```python
cnn = tf.keras.Sequential([
    tf.keras.Input(shape=(100, 100, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),       # anti-sobreajuste
    tf.keras.layers.Dense(2, activation='softmax'),
], name='cnn')
```

**Por qué**: las convoluciones detectan patrones locales (bordes = grietas) reutilizando filtros con pocos parámetros (Cuaderno 7, CIFAR-10). MaxPooling reduce resolución conservando señales dominantes. Dropout(0.5) apaga unidades al azar → regulariza con datos escasos.

## MobileNetV2 (transfer learning)

```python
base_mobilenet = tf.keras.applications.MobileNetV2(
    include_top=False, weights='imagenet', input_shape=(100, 100, 3))
base_mobilenet.trainable = False  # transfer learning: pesos congelados

mobilenet = tf.keras.Sequential([
    tf.keras.Input(shape=(100, 100, 3)),
    tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0),  # [0,1] -> [-1,1] (prepro MobileNetV2)
    base_mobilenet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation='softmax'),
], name='mobilenetv2')
```

**Por qué**: con ~408 imágenes, reutilizar pesos entrenados en ImageNet (millones de fotos) casi siempre supera a una CNN desde cero. Se congela el extractor de features y solo se entrena la cabeza densa. La capa `Rescaling` interna convierte [0,1] → [-1,1] **dentro del modelo**, así el contrato de datos (0-1) queda igual para los 3 modelos y la API no necesita cambios. Es la práctica estándar de producción para datasets pequeños.

**Nota sobre YOLO**: YOLO es para *detección* (encontrar dónde está el huevo), no clasificación. Este proyecto es un clasificador; la vía de YOLO queda como mejora futura para auto-detectar y recortar el huevo antes de clasificar.

## Comparación y selección

| Métrica | Rol |
|---|---|
| accuracy | Visión general |
| **recall `crack`** | **Criterio de selección** (error caro = huevo roto que pasa por bueno) |
| parámetros | Muestra por qué la CNN/MobileNet escalan mejor que el MLP |
| tiempo (s) | Justifica el uso de GPU T4 y el diseño |

El notebook genera: curvas loss/accuracy de los 3 modelos, classification report, matrices de confusión, 10 predicciones visuales y una tabla comparativa final con el ganador justificado (el que maximice el recall de `crack`).