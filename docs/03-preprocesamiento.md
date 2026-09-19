# 03 - Preprocesamiento de Imágenes

## Pipeline de Preprocesamiento (Paso a Paso)

El preprocesamiento es crítico: el modelo solo aprende de lo que le entregamos. Cada transformación tiene un propósito definido.

### 1. Bbox Crop con Margen del 2-5%

**Qué hace**: Recorta la imagen alrededor del bounding box detectado por Roboflow, añadiendo un pequeño margen porcentual.

**Por qué**:
- Elimina fondo irrelevante fuera de la zona del huevo
- Reduce variabilidad de entrada (menos fondo blanco/negro)
- Enfoca el modelo en la región de interés (la grieta o el área del huevo)
- Mejora el recall de `crack` al eliminar distracciones visuales

**Snippet de código**:

```python
import cv2
import numpy as np

def crop_with_margin(bbox, image, margin_pct=0.03):
    """
    Realiza crop alrededor del bbox con un margen porcentaje.
    
    bbox: [x_min, y_min, width, height] en formato COCO
    image: imagen completa (BGR, OpenCV)
    margin_pct: margen como porcentaje del bbox (0.03 = 3%)
    
    Retorna: imagen recortada con margen
    """
    x_min, y_min, w, h = bbox
    img_h, img_w = image.shape[:2]
    
    # Convertir a coordenadas absolutas
    x1 = max(0, int(x_min - margin_pct * w))
    y1 = max(0, int(y_min - margin_pct * h))
    x2 = min(img_w, int(x_min + w + margin_pct * w))
    y2 = min(img_h, int(y_min + h + margin_pct * h))
    
    cropped = image[y1:y2, x1:x2]
    return cropped
```

---

### 2. Conversión BGR → RGB

**Qué hace**: OpenCV lee imágenes en formato BGR por defecto; convertimos a RGB.

**Por qué**:
- La mayoría de modelos Keras/TensorFlow esperan RGB
- El dataset COCO original puede haber sido procesado en RGB
- Inconsistencia de canal causaría inferencia incorrecta si no se corrige

**Snippet de código**:

```python
import cv2

def bgr_to_rgb(image):
    """Convierte imagen de BGR a RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

---

### 3. Resize a 100×100

**Qué hace**: Redimensiona la imagen cropada a 100 × 100 píxeles cuadrados.

**Por qué**:
- **Tamaño de entrada fijo**: los modelos densos/convnet requieren dimensiones invariantes
- **Balance precisión/costo**: 100×100 es suficientemente grande para capturar características de grietas pero mantiene parámetros y tiempo de entrenamiento manejables
- **Consistencia**: todas las imágenes de entrada tienen exactamente el mismo tamaño

**Snippet de código**:

```python
import cv2

def resize_image(image, size=(100, 100)):
    """Redimensiona imagen al tamaño objetivo."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
```

---

### 4. Normalización /255

**Qué hace**: Divide todos los valores de píxel entre 255, escalando de [0, 255] a [0, 1].

**Por qué**:
- **Acelera convergencia**: gradients descent converge más rápido con valores en [0,1]
- **Estabilidad numérica**: evita que valores grandes dominen la función de pérdida
- **Standard practice**: práctica estándar para redes neuronales
- **Contrato de API**: la FastAPI espera valores normalizados en [0,1]

**Snippet de código**:

```python
import numpy as np

def normalize_image(image):
    """Normaliza imagen dividiendo entre 255."""
    return image / 255.0
```

---

### 5. Filtro Sobel Manual (Creatividad) — `cv2.filter2D`

**Qué aplica**: Gradiente de Sobel para resaltar bordes y grietas **antes** de que la imagen entrene la CNN.

**Por qué es creatividad**:
- **Feature engineering manual**: en lugar de dejar que la CNN aprendiera filtros desde cero, aplicamos un operador de borde clásico
- **Resalta grietas**: el operador Sobel calcula la primera derivada, resaltando áreas de cambio rápido de intensidad (grietas)
- **Muestra al proje intervención humana**: demuestra conocimiento de procesamiento de imágenes tradicional
- **Mejora el recall de crack**: estudios preliminares muestran que la entrada con Sobel resaltado mejora la detección de grietas finas

**Snippet de código**:

```python
import cv2

def apply_sobel(image):
    """
    Aplica filtro Sobel manual para resaltar grietas.
    
    Los pasos:
    1. Convertir a gray escala
    2. Aplicar Sobel en X y Y dirección
    3. Combinar magnitud de gradientes
    4. Volver a espacio RGB para compatibilidad con el pipeline
    """
    # 1. Escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Sobel en X y Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Magnitud del gradiente
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # 4. Normalizar a [0, 255] y convertir a uint8
    sobel_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_uint8 = np.uint8(sobel_norm)
    
    # 5. Volver a RGB (3 canales) para compatibilidad
    sobel_rgb = cv2.cvtColor(sobel_uint8, cv2.COLOR_GRAY2RGB)
    
    return sobel_rgb
```

**Flujo completo de preprocesamiento**:

```text
Imagen RAW → Bbox Crop (2-5% margen) → BGR→RGB → Resize 100×100 → /255 → [Opcional] Sobel cv2.filter2D → Modelo
```

**Cuándo usar Sobel**:
- **Siempre** para la comparación MLP vs CNN (demuestra la contribución del feature engineering manual)
- **Opcional** en producción si el recall de crack necesita maximización extra
- **Siempre documentado** en la predicción: mostrar al usuario que la grieta fue resaltada

---

*Cada paso de preprocesamiento está documentado con su "por qué" para facilitar la reproducción y la sustentación ante el profesor.*