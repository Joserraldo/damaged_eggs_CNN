# 04 - Modelos

Se entrenan dos arquitecturas con los MISMOS datos, aumentación y callbacks para una comparación justa.

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

## Comparación y selección

| Métrica | Rol |
|---|---|
| accuracy | Visión general |
| **recall `crack`** | **Criterio de selección** (error caro = huevo roto que pasa por bueno) |
| parámetros | Muestra por qué la CNN escala mejor |
| tiempo (s) | Justifica el uso de GPU T4 y el diseño |

El notebook genera: curvas loss/accuracy lado a lado, classification report, matrices de confusión, 10 predicciones visuales y una tabla comparativa final con el ganador justificado.