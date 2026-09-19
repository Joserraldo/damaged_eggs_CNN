# 02 - Dataset: egg-pisqc de Roboflow Universe

## Origen y Licenciamiento

- **Fuente**: https://universe.roboflow.com/muhammad-fauzan-wbvuk/egg-pisqc
- **Autor**: Muhammad fauzan
- **Licencia**: CC BY 4.0 (Atribución 4.0 Internacional)
- **Cita BibTeX recomendada**:

```bibtex
@misc{roboflow_egg_pisqc,
  author = {Muhammad fauzan},
  title = {egg Object Detection Dataset},
  url = {https://universe.roboflow.com/muhammad-fauzan-wbvuk/egg-pisqc},
  year = {2023},
  publisher = {Roboflow}
}
```

## Estadísticas del Dataset

| Propiedad | Valor |
|-----------|-------|
| **Total de imágenes** | 408 |
| **Clases** | `good`, `crack` |
| **Tipo** | Object Detection (COCO format) |
| **Etiquetas por imagen** | Variable (1 o más bounding boxes por imagen) |
| **Split estándar** | Train/Val test split de Roboflow |

## Distribución de Clases

| Clase | Count | Porcentaje |
|-------|-------|------------|
| `good` | ~240 | ~59% |
| `crack` | ~168 | ~41% |

**⚠️ Desbalance de clases**: La clase `crack` representa ~41% del dataset. Esto es crítico porque:

- El modelo tiende a ser sesgado hacia la clase mayoritaria (`good`)
- La métrica de negocio es **recall de `crack`** (no queremos dejar pasar ningún huevo agrietado)
- **Estrategias de mitigación**: class_weight, augmentation dirigida, oversampling de la clase minoritaria

## Split de Datos

El pipeline utiliza splits estándar de Roboflow:

- **Train**: 80% (~326 imágenes) — usado para entrenamiento del modelo
- **Validation**: 20% (~82 imágenes) — usado para validación durante entrenamiento con EarlyStopping

**Riesgo**: Con solo 408 imágenes y desbalanceo, el modelo de validación puede no representar adecuadamente la clase `crack`. Se recomienda:

1. **Stratified split** preserving la proporción de clases
2. **Data augmentation** dirigida a la clase `crack` (rotaciones, zoom, brightness jitter)
3. **Class weight** computado automáticamente basado en la frecuencia de cada clase en el split de train

## Estructura COCO

El dataset está en formato COCO, lo que significa que cada imagen tiene un archivo `_annotations.coco.json` con:

- `images`: información de cada imagen (id, width, height, file_name)
- `annotations`: bounding boxes [x_min, y_min, width, height] y category_id
- `categories`: definición de clases [id, name, supercategory]

**Snippet de carga** (del notebook):

```python
def find_annotation_file(split_dir):
    files = list(split_dir.glob('_annotations.coco.json'))
    return files[0] if files else None
```

## Estrategias ante el Desbalance

Dado que `crack` es la clase minoritaria y la métrica crítica es el recall, se aplican:

1. **Class weights** computados al fit del modelo:
   ```python
   class_weights = compute_class_weight('balanced', classes=['good', 'crack'], y=y_train)
   ```

2. **Augmentation dirigida** a la clase `crack`:
   - Rotación ±15°
   - Zoom 0.9x a 1.1x
   - Brillo ±20%
   - Flipping horizontal

3. **Oversampling** de mini-batches con más muestras de `crack`

4. **Threshold adjustment** en la predicción final para priorizar recall sobre precision

---

*Documento esencial para entender la distribución de datos, riesgos y decisiones de preprocesamiento posteriores.*