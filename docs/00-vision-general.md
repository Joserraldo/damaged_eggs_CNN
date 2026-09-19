# 00 - Visión General del Proyecto

## Propuesta de Valor

**Egg Quality** es una solución de aprendizaje automático que clasifica automáticamente si un huevo se encuentra en buen estado (`good`) o presenta indicios de agrietamiento (`crack`). El proyecto combina procesamiento digital de imágenes, redes neuronales densas y redes neuronales convolucionales para construir un flujo completo: desde la preparación del dataset hasta una aplicación móvil capaz de enviar imágenes a un modelo entrenado.

El objetivo no es reemplazar una inspección especializada, sino crear una herramienta de apoyo que permita automatizar una primera revisión de calidad de forma rápida, reproducible y accesible desde un celular.

## Problema

Los huevos agrietados representan un costo significativo en la cadena de producción y distribución:
- **Pérdida económica**: huevos agrietados no aptos para venta comercial
- **Riesgo sanitario**: grietas pueden permitir entrada de bacterias
- **Inspección manual lenta y subjetiva**: los métodos actuales dependen de la vista humana, son lentos y propensos a errores

## Usuario

- **Principal**: Empacadoras y distribuidores de huevos que necesitan filtrar huevos de mala calidad automáticamente
- **Secundario**: Consumidores finales que buscan garantizar la calidad del producto
- **Demo en vivo**: Aplicación móvil que captura una foto y en tiempo real muestra si el huevo es `good` o `crack` con explicación breve

## Alcance

- **Dataset**: Roboflow `egg-pisqc` (408 imágenes, 2 clases: `good`/`crack`, object detection COCO, licencia CC BY 4.0)
- **Preprocesamiento**: Bbox crop con margen, resize 100×100 RGB, normalización /255, filtro Sobel manual opcional
- **Modelos comparados**: MLP (Flatten→Dense) vs CNN (Conv2D + MaxPool + Dropout)
- **Mejor métrica**: Recall de `crack` (es más costoso dejar pasar un huevo agrietado que marcar uno bueno para revisión)
- **Pipeline**: Google Colab (GPU T4) → modelo .keras → FastAPI → Expo Go (aplicación móvil)
- **Despliegue**: Notebook → Google Drive → zip → VM → FastAPI → aplicación móvil

## Creatividad Destacada

1. **Filtro Sobel manual con cv2.filter2D** aplicado antes de la CNN para resaltar grietas
2. **Comparativa MLP vs CNN** con curvas de entrenamiento lado a lado y matrices de confusión por clase
3. **Histograma de normalización** que muestra la diferencia visual entre datos sin normalizar vs /255
4. **Predicción visual con 10 ejemplos test** etiquetados como `good` (verde) o `crack` (rojo)
5. **Tabla comparativa final** que resume accuracy, recall crack, parámetros y tiempo de entrenamiento

---

*Documento base para toda la documentación técnica del proyecto. Cada decisión posterior está justificada en relación con esta visión.*