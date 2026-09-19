# 05 - Creatividad (features destacables para la sustentación)

El profe valora **creatividad**, uso de GPU de Colab y documentación del **PORQUÉ** de cada decisión. Esta lista es el guion de lo destacable en la demo.

## 1. ⭐ Filtro Sobel manual antes de la CNN
**Qué**: celda que aplica `cv2.Sobel` + magnitud para resaltar bordes, mostrando el mosaico `original → crop → /255 → sobel`.
**Qué demuestra**: dominio de procesamiento clásico (Cuaderno 6) y que entendemos que la CNN "aprende" exactamente ese tipo de filtro en sus primeras capas. La grieta ES un borde de alto gradiente.

## 2. ⭐ Comparativa justa MLP vs CNN
**Qué**: mismas particiones, misma aumentación, mismos callbacks; solo cambia la arquitectura.
**Qué demuestra**: rigor experimental — no es "mi red gana", es "con condiciones idénticas, la convolución aporta X".

## 3. ⭐ Recall de crack como criterio de negocio
**Qué**: la selección del mejor modelo usa `recall_crack`, no accuracy; se explica con la tabla de costos asimétricos (falso negativo caro).
**Qué demuestra**: pensamiento de aplicación real, no solo métricas de juguete.

## 4. ⭐ Curvas de entrenamiento lado a lado + sobreajuste
**Qué**: `plot_history` con loss/accuracy y val en ambas, detectando overfitting y explicando las medidas (early stopping, dropout, augmentation).
**Qué demuestra**: lectura crítica de resultados y conocimiento de regularización.

## 5. ⭐ Predicción visual en test (10 ejemplos)
**Qué**: grid con imagen real, etiqueta verdadera, predicción y confianza; verde = acierto, rojo = error.
**Qué demuestra**: comunicación visual del resultado — lo que el profe recuerda de una demo.

## 6. ⭐ Matrices de confusión + classification report por clase
**Qué**: ConfusionMatrixDisplay ('Blues') y reporte por clase para ambos modelos.
**Qué demuestra**: análisis por clase (no solo promedio), y la diferencia good vs crack hecha explícita.

## 7. ⭐ Contrato de preprocesamiento documentado y replicado en la API
**Qué**: `preprocesamiento.json` (crop 100x100, /255, RGB, margin 0.04) exportado con el modelo y reproducido en `api/main.py`.
**Qué demuestra**: ingeniería de fin a fin — el modelo no vive solo, vive en un sistema con un contrato.

## 8. ⭐ GPU T4 explícita y tiempo medido
**Qué**: `!nvidia-smi` + `time.time()` por modelo y comparativa de duración.
**Qué demuestra**: uso real de la GPU de Colab y justificación del diseño (100x100 cabe en T4; 2-4h presupuestadas).