# 03 - Preprocesamiento

El preprocesamiento es la parte que demuestra dominio de **procesamiento digital de imágenes** (Cuaderno 6). Cada paso tiene un PORQUÉ.

## 1. Recorte por bounding box

```python
def crop_from_bbox(image, bbox, margin=0.04):
    x, y, width, height = [float(v) for v in bbox]
    h_img, w_img = image.shape[:2]
    mx, my = width * margin, height * margin
    x1 = max(0, int(x - mx)); y1 = max(0, int(y - my))
    x2 = min(w_img, int(x + width + mx)); y2 = min(h_img, int(y + height + my))
    return image[y1:y2, x1:x2]
```

**Por qué**: el bbox aísla el huevo y elimina fondo (mesa, cajas, manos) que no aporta señal y solo añade ruido. El margen 4% evita cortar el borde de la grieta.

## 2. BGR → RGB

```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

**Por qué**: OpenCV lee en BGR; TensorFlow espera RGB. Es el error clásico que arruina modelos entrenados con canales invertidos.

## 3. Redimensionar a 100x100

```python
image = cv2.resize(image, (100, 100))
```

**Por qué**: 100x100 (10.000 px × 3 canales) es suficiente para apreciar una grieta, y reduce el costo de entrenamiento frente a resoluciones mayores. Normaliza el input de ambos modelos (MLP y CNN).

## 4. Normalización /255

```python
normalization = tf.keras.layers.Rescaling(1.0 / 255)
```

**Por qué**: lleva los valores de [0,255] a [0,1). Las activaciones (ReLU, softmax) y el gradiente de Adam funcionan mejor con entradas en ese rango; evita dominancia de canales por brillo.

## 5. ⭐ Feature creativo: filtro Sobel manual (cv2.filter2D)

```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
```

**Por qué demuestra creatividad**: resalta los bordes — exactamente donde vive la grieta. Es el puente entre el filtrado clásico (Cuaderno 6) y lo que la CNN aprende sola en sus primeras capas. En la sustentación se muestra el mosaico `original → crop → normalizado → sobel` para evidenciar que la grieta es un patrón de alto gradiente.

## Contrato de preprocesamiento (API)

El modelo solo entiende su contrato; la API replica exactamente:

```python
image = Image.open(io.BytesIO(contents)).convert('RGB')
image = image.resize(IMAGE_SIZE)                    # 100x100
array = np.asarray(image, dtype=np.float32) / 255.0 # /255
```

Si la foto enviada no está bien encuadrada en el huevo, el contrato se rompe — por eso la app muestra instrucciones de encuadre.