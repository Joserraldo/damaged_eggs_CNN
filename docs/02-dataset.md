# 02 - Dataset

## Fuente

- **Dataset**: egg Object Detection Dataset (`egg-pisqc`)
- **Autor**: Muhammad Fauzan (Roboflow Universe)
- **URL**: https://universe.roboflow.com/muhammad-fauzan-wbvuk/egg-pisqc
- **Tipo**: detección de objetos (bounding boxes)
- **Imágenes**: 408 · **Clases**: `good`, `crack`
- **Versiones**: 2 · **Licencia**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Cita (BibTeX)

```bibtex
@misc{ egg-pisqc_dataset,
  title = { egg Dataset },
  type = { Open Source Dataset },
  author = { Muhammad fauzan },
  howpublished = { \url{ https://universe.roboflow.com/muhammad-fauzan-wbvuk/egg-pisqc } },
  url = { https://universe.roboflow.com/muhammad-fauzan-wbvuk/egg-pisqc },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2023 },
  month = { jul },
  note = { visited on 2026-09-19 },
}
```

## Formato COCO

Roboflow descarga en formato COCO: cada split (`train/valid/test`) contiene `_annotations.coco.json` con las bounding boxes de cada huevo. El notebook usa esas cajas para generar los crops de entrenamiento.

| Campo COCO | Uso |
|---|---|
| `images[].file_name` | Ubicar la imagen original |
| `annotations[].bbox` | Coordenadas `[x, y, w, h]` del huevo |
| `categories[].name` | Mapear id → `good` / `crack` |

## Balance de clases y riesgo

Con solo 408 imágenes el riesgo principal es el sobreajuste y un posible desbalance entre clases. El notebook reporta el conteo por split y propone dos estrategias:

1. **class_weight** en `model.fit`: pondera más la clase minoritaria en la pérdida.
2. **Augmentation dirigida**: `RandomFlip`, `RandomRotation(0.08)`, `RandomZoom(0.10)` — genera variantes sintéticas y reduce el sobreajuste con pocos datos.

## Por qué el recall de crack es la métrica de negocio

Costo asimétrico:

| Real \ Predicho | good | crack |
|---|---|---|
| **good** | ✓ OK | Revisión extra (costo bajo) |
| **crack** | **ERROR CARO** ✗ | ✓ OK |

Dejar pasar un huevo agrietado (falso negativo de `crack`) es el peor error: el producto defectuoso sale a la venta. Por eso la selección del mejor modelo privilegia `recall` de `crack` sobre accuracy.