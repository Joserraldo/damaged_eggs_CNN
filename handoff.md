# Handoff - Egg Quality

Fecha: 2026-09-19
Estado: listo para ejecutar en Colab (notebook con pip install, augmentation aplicada y transfer learning) + deploy AWS documentado + app con cámara en vivo; quedan: ejecutar el entrenamiento real y subir el modelo a AWS.
Última revisión: 2026-09-19

## Objetivo

Construir una aplicación que permita tomar una foto o seleccionar una imagen de un huevo desde el celular y clasificarla como:

- `good`: huevo aparentemente bueno.
- `crack`: huevo posiblemente agrietado.

La métrica más importante será el recall de `crack`, porque es más costoso dejar pasar un huevo agrietado que marcar uno bueno para revisión.

## Fuente del dataset

Se utiliza el dataset público **egg Object Detection Dataset** de Roboflow Universe:

- Fuente: https://universe.roboflow.com/muhammad-fauzan-wbvuk/egg-pisqc
- Autor indicado por Roboflow: Muhammad fauzan
- 408 imágenes y 2 clases: `good` y `crack`
- Tipo: detección de objetos, con anotaciones de bounding boxes
- Licencia indicada por la fuente: CC BY 4.0

El notebook descarga el dataset en formato COCO mediante la API de Roboflow y usa sus bounding boxes para generar los recortes de entrenamiento.

## Arquitectura actual

```text
Expo Go en el celular
        |
        | HTTP multipart/form-data (IP elástica AWS, puerto 8000)
        v
FastAPI en EC2 (AWS)
        |
        | TensorFlow / Keras
        v
mejor_*.keras (MLP | CNN | MobileNetV2)
```

## Lo que ya se hizo

### Documentación técnica (nueva)

Carpeta: `docs/` — 7 documentos que convierten el esqueleto en una base evaluable y con foco en creatividad:

- `00-vision-general.md`: propuesta de valor, problema, usuario y alcance.
- `01-arquitectura.md`: diagrama Mermaid, flujo de datos y tabla de decisiones con su PORQUÉ.
- `02-dataset.md`: fuente Roboflow, licencia CC BY 4.0, formato COCO, balance y por qué manda el recall de `crack`.
- `03-preprocesamiento.md`: bbox crop, 100x100, /255, BGR→RGB y el filtro Sobel manual (`cv2.filter2D`) como feature creativo.
- `04-modelos.md`: arquitecturas MLP vs CNN, callbacks justificados y criterio de selección.
- `05-creatividad.md`: guion de 8 features destacables para la sustentación (Sobel, comparativa justa, recall de negocio, curvas, predicción visual...).
- `06-deploy-vm.md`: notebook → Drive → zip → VM → FastAPI, con el contrato de preprocesamiento.

### Notebook de entrenamiento

Archivo: `clasificador_huevos_colab.ipynb` — **reparado y actualizado el 2026-09-19 con 3 mejoras**:

1. **`pip install` en la primera celda** (roboflow, tensorflow, opencv, scikit-learn, pandas, matplotlib) — idempotente, por si Colab no trae algo.
2. **Augmentation realmente aplicada**: el código definía la capa de rotación/flip/zoom pero nunca la conectaba a `train_ds`. Ahora se aplica SOLO en train (nunca valid/test).
3. **Tercer modelo con transfer learning (MobileNetV2)**: pesos de ImageNet congelados, solo la cabeza densa entrena, `Rescaling` interna [0,1]→[-1,1] para no romper el contrato de datos 0-1. La selección del mejor modelo ahora es entre MLP, CNN y MobileNetV2 por recall de `crack`.

Se reconstruyó el flujo completo en 23 celdas:

- Celda 0: `pip install` de todas las dependencias por si acaso.
- Celdas 1-15: configuración, GPU, descarga COCO, preprocesamiento por bbox, mosaicos, balance, carga de datos con `/255` y augmentation aplicada solo a train.
- Celda 16: definición de MLP, CNN y MobileNetV2 + entrenamiento con callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) y medición de tiempo.
- Celdas 17-19: evaluación comparativa — curvas de los 3 modelos, classification report, matrices de confusión, selección del mejor modelo por recall de `crack`, 10 predicciones visuales y tabla comparativa.
- Celdas 20-22: exportación a Drive (`egg_quality_models.zip`) e instrucciones para la VM AWS.

Incluye el filtro Sobel manual (`cv2.filter2D`) ANTES de la CNN como demostración de procesamiento clásico.

**Sigue pendiente**: ejecutar el notebook de principio a fin en Colab con GPU T4 y la API key real de Roboflow.

### Frontend móvil

Carpeta: `mobile/`

Archivos principales:

- `App.js`: pantalla de clasificación.
- `package.json`: dependencias de Expo.
- `app.json`: configuración de la aplicación.

La pantalla ya contempla:

- **Cámara en vivo**: preview de video continuo (`expo-camera`) con botón de captura manual del fotograma.
- Selección desde galería.
- Captura con foto.
- Vista previa de la imagen.
- Estado de carga.
- Mensajes de error.
- Resultado visual para `good` o `crack`.
- Porcentaje de confianza.
- Configuración visual propia, no solo una pantalla genérica.

La URL de la API se configura en la app (botón ⚙️) o en `mobile/App.js`; para la demo usar la **IP elástica de la VM AWS** y el puerto 8000 (ver `docs/06-deploy-vm.md`).

### Backend

Carpeta: `api/`

Archivos principales:

- `main.py`: API FastAPI.
- `requirements.txt`: dependencias Python.

Endpoints actuales:

- `GET /health`: indica si la API está viva y si encontró el modelo.
- `POST /predict`: recibe una imagen en el campo `file` y devuelve etiqueta, confianza y probabilidades por clase.

El backend busca `mejor_cnn.keras` junto a `main.py`, o la ruta indicada con la variable de entorno `MODEL_PATH`.

### Documentación inicial

- `README.md`: instrucciones generales para ejecutar API y aplicación móvil.
- `handoff.md`: este documento, con el estado del proyecto, decisiones técnicas y pendientes.
- `.gitignore`: ya creado; excluye secretos, modelos, datasets, dependencias generadas y `.opencode/`.

## Lo que está pendiente (para mañana)

### 1. Ejecutar el notebook en Colab (la tarea principal)

1. Abrir `clasificador_huevos_colab.ipynb` en Google Colab.
2. Entorno de ejecución → Cambiar tipo → **GPU T4**.
3. Ejecutar las 23 celdas en orden.
4. En la celda 4, pegar la API key de Roboflow (https://app.roboflow.com/settings/api) — solo durante la ejecución, nunca guardarla en el archivo.
5. Verificar: GPU visible en `nvidia-smi`, versión de Roboflow descargada, crops generados en `train/valid/test`, balance de clases.
6. Confirmar que MLP, CNN y MobileNetV2 entrenan y el ganador se elige por recall de `crack`.
7. Confirmar `egg_quality_models.zip` en Drive (`MyDrive/egg_quality/`).

El notebook usa rutas habituales de Roboflow, pero la estructura real descargada debe confirmarse en Colab. No presentar el entrenamiento como validado hasta ejecutarlo completo.

### 2. Subir el modelo a AWS

Guía completa en `docs/06-deploy-vm.md`. Resumen:

1. EC2 → Ubuntu 22.04, **t3.medium** (TF necesita >1 GB RAM), security group con SSH (22, a tu IP) y **Custom TCP 8000 → 0.0.0.0/0**.
2. **IP elástica** → asociar a la instancia (la IP no cambia al apagar).
3. `scp -i llave.pem egg_quality_models.zip ubuntu@IP:/home/ubuntu/` y descomprimir.
4. Clonar el repo (o scp de `api/`), crear venv, `pip install -r requirements.txt`.
5. Servicio systemd `egg-quality.service` (plantilla en docs/06) con `MODEL_PATH` al modelo ganador.
6. Verificar `http://IP_ELASTICA:8000/health` → `model_loaded: true`.

### 3. Probar la app desde el celular contra AWS

1. `cd mobile && npm install && npx expo start` (o botón ⚙️ en la app).
2. Configurar URL: `http://IP_ELASTICA:8000`.
3. Probar las 3 fuentes: **En vivo** (captura manual del preview), **Foto** y **Galería**.
4. Probar: respuesta correcta, imagen inválida, API apagada.
5. Revisar permisos de cámara en Android real.

### 4. Antes de sustentar

- Confirmar orden de clases `['good', 'crack']` en la API vs el modelo exportado (`clases.json`).
- Probar imágenes conocidas de ambas clases contra `/predict`.
- No subir la API key de Roboflow ni el modelo pesado al repo (`.gitignore` ya lo cubre).
- Opcional: YOLO para auto-detectar y recortar el huevo antes de clasificar (hoy solo se clasifica si el huevo viene bien encuadrado).

## Preparación para GitHub

La preparación inicial ya está hecha en `.gitignore`. Antes de publicar el repositorio todavía falta:

1. No subir la API key de Roboflow.
2. No subir imágenes privadas del dataset si la licencia o el proyecto no lo permiten.
3. Decidir si el modelo se publica en GitHub, Git LFS, Drive o un almacenamiento externo.
4. Revisar que `README.md` explique instalación, ejecución, fuente del dataset y conexión móvil.
5. Inicializar Git y revisar `git status` antes del primer commit.
6. Crear el repositorio remoto en GitHub.
7. Hacer el primer commit solo cuando la estructura esté revisada.

Comandos previstos, para ejecutar cuando se decida publicar:

```powershell
git init
git add .
git status
git commit -m "feat: create egg quality app foundation"
git branch -M main
git remote add origin https://github.com/USUARIO/egg-quality.git
git push -u origin main
```

Estos comandos todavía no se han ejecutado.

## Estado resumido

| Área | Estado |
|---|---|
| Notebook escrito (23 celdas, 3 modelos, augmentation, pip install) | Hecho, pendiente de ejecución completa en Colab |
| Fuente y licencia del dataset | Documentadas |
| GPU y Roboflow | Escrito, falta probar |
| Preprocesamiento | Escrito, falta validar rutas reales |
| Entrenamiento MLP | Escrito, falta ejecutar |
| Entrenamiento CNN | Escrito, falta ejecutar |
| Transfer learning MobileNetV2 | Escrito, falta ejecutar |
| Selección del mejor modelo | Escrito (3 modelos, criterio recall crack), falta ejecutar |
| API FastAPI | Escrita y con 6 tests pasando |
| Frontend Expo (cámara en vivo + foto + galería) | Escrito |
| Deploy AWS (EC2, IP elástica, puerto 8000, systemd) | Documentado, falta ejecutar |
| Conexión con modelo real | Pendiente (requiere ejecutar Colab) |
| Prueba desde celular contra AWS | Pendiente |
| Detector automático de huevo (YOLO) | Pendiente opcional |
| `.gitignore` y exclusión de `.opencode/` | Hecho |
| GitHub | Pendiente |

## Próximo orden recomendado

1. Abrir el notebook en Colab con GPU T4 y ejecutarlo de principio a fin (API key de Roboflow al llegar a la celda 4).
2. Confirmar el modelo ganador y `egg_quality_models.zip` en Drive.
3. Crear la EC2 en AWS (Ubuntu, t3.medium) + IP elástica + puerto 8000 abierto (docs/06-deploy-vm.md).
4. Subir el zip a la VM (`scp`), instalar dependencias de `api/` y dejar `uvicorn` con systemd.
5. Verificar `http://IP_ELASTICA:8000/health` desde el celular.
6. Conectar Expo Go desde el celular apuntando a la IP elástica (botón ⚙️ en la app).
7. Probar cámara en vivo, foto y galería; corregir problemas de red, permisos o formato.
8. Revisar secretos y subir la primera versión a GitHub.
