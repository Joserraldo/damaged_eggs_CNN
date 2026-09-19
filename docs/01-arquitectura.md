# 01 - Arquitectura del Sistema

## Diagrama de Flujo

```mermaid
flowchart TD
    %% Dataset y Origen
    DS[Roboflow egg-pisqc Dataset<br/>(408 img, COCO, 2 clases)] -->|Descarga y split| GC[Google Colab GPU T4]

    %% Preprocesamiento
    GC -->|Bbox crop + margen 2-5%| PC[Preprocesamiento de Imágenes]
    PC -->|BGR→RGB + resize 100×100 + /255| NC[Normalización]

    %% Sobel (creatividad)
    NC -->|Sobel cv2.filter2D (opcional)| SO[Filtro Sobel Manual]

    %% Modelo
    SO -->|MLP o CNN| MT[Entrenamiento de Modelo]
    MT -->|Saving| MK[modelo .keras]

    %% Despliegue
    MK -->|gdown/zip| VM[VM Servidor FastAPI]
    VM -->|API REST| API[FastAPI Server]

    %% Aplicación Móvil
    API -->|HTTP POST JSON| APP[Expo Go / App.js]
    APP -->|Resultado en tiempo real| USER[Usuario Móvil]
```

## Decisiones y PORQUÉ de cada capa

| Etapa | Decisión | Justificación (Por qué) |
|-------|----------|-------------------------|
| **Dataset → Colab** | Usar Roboflow Universe | Dataset público accesible, licenciamiento CC BY 4.0, 408 imágenes suficientes para proyecto académico con 2 clases balanceadas(ish) |
| **Colab GPU T4** | Entrenamiento en Colab | Acceso gratuito a GPU T4, entorno listo para Jupyter/.ipynb, compatibilidad con TensorFlow/Keras |
| **Bbox crop con margen** | Recortar alrededor de la grieta detectada por Roboflow | Elimina fondo irrelevante, enfoca el modelo en la zona de interés, reduce variabilidad de entrada |
| **BGR→RGB + resize 100×100** | Convertir y redimensionar consistente | Los modelos Keras esperan RGB, el tamaño fijo (100×100) equilibra complejidad vs precisión para imágenes de huevo |
| **Normalización /255** | Escalar píxeles a [0,1] | Acelera convergencia de gradientes, evita que valores de pixel grandes (0-255) dominen la función de pérdida |
| **Sobel cv2.filter2D (creatividad)** | Realzar bordes y grietas antes de CNN | **Feature engineering manual**: resalta patrones de grietas que la CNN aprendería de todas formas, pero muestra al profe intervención humana explícita y conocimiento de procesamiento de imágenes |
| **MLP vs CNN** | Comparar arquitecturas | MLP: baseline rápido, arquitectura densa simple. CNN: extrae características espaciales, detectores de bordes locales. El recall de crack es la métrica de negocio, y CNN históricamente obtiene mejor recall en visión por computador |
| **Guardar .keras** | Formato nativo TensorFlow | Soporta saving/loading completo, incluye optimizer state, compatible con TensorFlow Serving y Flask/FastAPI |
| **FastAPI servidor** | API REST ligera | Performance alta (async), documentación automática, integración fácil con Expo Go, dependency minimal |
| **Expo Go móvil** | Aplicación cliente | Cross-platform (iOS/Android), recarga rápida, usa cámara nativa, comunicación HTTP con API local/cloud |
| **Recall crack prioritario** | Métrica de negocio | Evitar falsos negativos (marcar crack como good) es más costoso que falsos positivos (revisar huevo bueno) |

## Contrato de Imagen

- **Formato**: RGB (3 canales)
- **Dimensiones**: 100 × 100 píxeles
- **Rango de valores**: [0, 1] (después de /255)
- **Orden de canales**: RGB (no BGR, que es el default de OpenCV)
- **Preprocesamiento obligatorio**: Crop bbox → BGR→RGB → Resize → /255 → Opcional: Sobel

---

*Cada decisión de arquitectura está documentada con su "por qué" para facilitar la sustentación y futuras mejoras.*