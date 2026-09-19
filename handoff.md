# Handoff - Egg Quality

Fecha: 2026-09-19
Estado: base de aplicación creada; entrenamiento e integración real del modelo pendientes.
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
        | HTTP multipart/form-data
        v
FastAPI en el computador o VM
        |
        | TensorFlow / Keras
        v
mejor_cnn.keras
```

## Lo que ya se hizo

### Notebook de entrenamiento

Archivo: `clasificador_huevos_colab.ipynb`

Incluye la estructura para:

- Configurar Google Colab y verificar GPU.
- Solicitar la API key de Roboflow con `getpass`.
- Montar Google Drive.
- Descargar el proyecto `egg-pisqc` en formato COCO.
- Leer anotaciones y recortar imágenes por bounding box.
- Añadir margen, convertir BGR a RGB y redimensionar a `100x100`.
- Visualizar imágenes originales, recortes, normalización y filtro Sobel.
- Crear mosaicos y revisar el balance de clases.
- Cargar los crops con TensorFlow.
- Normalizar imágenes con `/255`.
- Aplicar augmentation.
- Entrenar un MLP y una CNN.
- Calcular accuracy, recall de `crack`, classification report y matrices de confusión.
- Elegir el mejor modelo y exportarlo a Drive.

El notebook es una plantilla escrita, pero todavía no se ha ejecutado de principio a fin en Colab.

### Frontend móvil

Carpeta: `mobile/`

Archivos principales:

- `App.js`: pantalla de clasificación.
- `package.json`: dependencias de Expo.
- `app.json`: configuración de la aplicación.

La pantalla ya contempla:

- Selección desde galería.
- Captura con cámara.
- Vista previa de la imagen.
- Estado de carga.
- Mensajes de error.
- Resultado visual para `good` o `crack`.
- Porcentaje de confianza.
- Configuración visual propia, no solo una pantalla genérica.

La URL de la API está definida en `mobile/App.js` y debe cambiarse por la IPv4 del computador donde se ejecute FastAPI.

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

## Lo que está pendiente

### 1. Ejecutar y depurar el notebook

Pendiente principal:

1. Abrir el notebook en Google Colab.
2. Activar GPU T4 si está disponible.
3. Ejecutar las celdas en orden.
4. Introducir la API key de Roboflow solo durante la ejecución.
5. Confirmar que Roboflow descarga la versión esperada.
6. Revisar las rutas reales que devuelve el dataset COCO.
7. Confirmar que se generan crops en `train`, `valid` y `test`.
8. Revisar el balance de clases.
9. Entrenar MLP y CNN.
10. Comparar especialmente el recall de `crack`.
11. Guardar el mejor modelo y descargar `mejor_cnn.keras` o el modelo ganador.

El notebook usa rutas y nombres habituales de Roboflow, pero la estructura real descargada debe confirmarse en Colab antes de considerarlo cerrado. No se debe presentar el entrenamiento como validado hasta ejecutar el notebook completo en Colab.

### 2. Resolver el contrato de imagen

El entrenamiento usa imágenes recortadas por bounding box. La primera API toma la imagen recibida, la convierte a RGB, la redimensiona a `100x100` y aplica `/255`.

Esto funciona correctamente solo si la foto enviada ya está bien encuadrada en el huevo. Para una versión más robusta hay dos opciones:

- Mantener la regla de que el usuario encuadre un solo huevo y documentarlo en la app.
- Entrenar o integrar un detector que encuentre automáticamente el huevo y recorte su bounding box antes de clasificar.

La segunda opción es mejor para producción, pero requiere otro modelo o un pipeline de detección.

### 3. Conectar el modelo real

Después del entrenamiento:

1. Copiar el archivo del modelo a `api/mejor_cnn.keras`.
2. Comprobar que el backend devuelve `model_loaded: true` en `/health`.
3. Confirmar que el orden de clases coincide con `['good', 'crack']`.
4. Confirmar que el modelo espera imágenes RGB de `100x100` normalizadas.
5. Probar varias imágenes conocidas de ambas clases.

No se debe subir la API key de Roboflow ni datos privados al repositorio.

### 4. Probar Expo Go

Pendiente:

1. Instalar Node.js y dependencias de `mobile`.
2. Ejecutar Expo.
3. Obtener la IPv4 local con `ipconfig`.
4. Cambiar `API_URL` en `mobile/App.js`.
5. Mantener celular y computador en la misma red Wi-Fi.
6. Abrir la app con Expo Go.
7. Probar galería y cámara.
8. Revisar errores de permisos de cámara.
9. Probar una respuesta correcta, una imagen inválida y la API apagada.

Si la red Wi-Fi bloquea conexiones entre dispositivos, se necesitará otra solución de túnel o una VM accesible desde el celular.

### 5. Mejorar el backend antes de producción

Pendiente recomendable:

- Validar tamaño máximo de archivo.
- Añadir logging controlado.
- Separar configuración en variables de entorno.
- No dejar `allow_origins=['*']` en producción.
- Añadir endpoint o mecanismo para cargar el modelo sin reiniciar, solo si realmente se necesita.
- Añadir pruebas para `/health` y `/predict`.
- Agregar una respuesta de versión del modelo.

### 6. Mejorar la app móvil

Pendiente recomendable:

- Añadir instrucciones visibles para encuadrar el huevo.
- Mostrar el resultado con una explicación breve, no como diagnóstico definitivo.
- Añadir historial local de análisis si el proyecto lo requiere.
- Añadir una pantalla de configuración para cambiar la URL de API sin editar código.
- Revisar permisos y comportamiento en Android real.
- Añadir icono y splash definitivos.

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
| Notebook escrito | Hecho como plantilla, pendiente de ejecución completa |
| Fuente y licencia del dataset | Documentadas |
| GPU y Roboflow | Escrito, falta probar |
| Preprocesamiento | Escrito, falta validar rutas reales |
| Entrenamiento MLP | Escrito, falta ejecutar |
| Entrenamiento CNN | Escrito, falta ejecutar |
| Selección del mejor modelo | Escrito, falta ejecutar |
| API FastAPI | Base escrita |
| Frontend Expo | Base escrita |
| Conexión con modelo real | Pendiente |
| Prueba desde celular | Pendiente |
| Detector automático de huevo | Pendiente opcional |
| Tests | Pendiente |
| `.gitignore` y exclusión de `.opencode/` | Hecho |
| GitHub | Pendiente |

## Próximo orden recomendado

1. Ejecutar el notebook en Colab y obtener el modelo.
2. Revisar el contrato de preprocesamiento con imágenes reales.
3. Copiar el modelo a `api/` y probar el endpoint.
4. Conectar Expo Go desde el celular.
5. Corregir problemas de red, permisos o formato.
6. Revisar secretos y verificar el contenido que entrará al primer commit.
7. Subir la primera versión a GitHub.
8. Después, mejorar el detector automático y la experiencia de usuario.
