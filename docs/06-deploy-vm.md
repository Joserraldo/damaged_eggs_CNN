# 06 - Deploy en AWS (EC2 + IP elástica)

Guía completa para llevar el modelo entrenado en Colab hasta una VM EC2 de AWS, y dejar la API visible desde el celular mediante la **IP elástica** y el **puerto 8000 abierto**.

## 1. En Colab (el notebook lo hace solo)

- Guarda el mejor modelo en `/content/mejor_*.keras` (el ganador entre MLP, CNN y MobileNetV2).
- Exporta a Drive: `MyDrive/egg_quality/models/` + `clases.json` + `preprocesamiento.json`.
- Comprime todo en `egg_quality_models.zip` y lo copia a Drive.

## 2. Crear la VM en AWS (EC2)

1. Consola AWS → EC2 → **Launch instance**.
2. Nombre: `egg-quality-api`. Imagen: **Ubuntu 22.04 LTS** (64-bit x86).
3. Tipo de instancia: **t3.medium** (2 vCPU / 4 GB RAM). TensorFlow carga con menos de 2 GB, pero 1 GB (t2.micro) puede hacer swap; medium es lo cómodo.
4. Par de llaves: crea o selecciona tu `.pem` (guarda la ruta).
5. Security group: crea uno nuevo `egg-quality-sg` con reglas de entrada:
   - **SSH** (22) → tu IP
   - **Custom TCP 8000** → `0.0.0.0/0` (aquí vive la API; la app móvil se conecta por este puerto)
6. Almacenamiento: **16 GB** (TensorFlow + dependencias ocupan ~4 GB).
7. Launch.

## 3. IP elástica (para que la IP no cambie al apagar la VM)

1. EC2 → **Elastic IPs** → Allocate Elastic IP address.
2. Selecciona la IP → **Actions → Associate Elastic IP address** → elige la instancia `egg-quality-api`.
3. Anota la IP (ej. `54.210.100.25`). Esta es la URL que usará la app: `http://54.210.100.25:8000`.

## 4. Subir el modelo a la VM

Opción A (recomendada, directa desde tu PC):

```powershell
scp -i "tu_llave.pem" egg_quality_models.zip ubuntu@54.210.100.25:/home/ubuntu/
```

Opción B (desde Drive en la VM):

```bash
pip install gdown
gdown <FILE_ID_de_egg_quality_models.zip>
```

Luego en la VM:

```bash
unzip egg_quality_models.zip -d egg_quality_models
```

## 5. Instalar dependencias y subir la API

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
git clone https://github.com/USUARIO/damaged_eggs_CNN.git   # o scp de la carpeta api/
cd damaged_eggs_CNN/api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Servicio persistente con systemd

`/etc/systemd/system/egg-quality.service`:

```ini
[Unit]
Description=Egg Quality API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/damaged_eggs_CNN/api
Environment=MODEL_PATH=/home/ubuntu/egg_quality_models/mejor_cnn.keras
Environment=MODEL_VERSION=v1.0
ExecStart=/home/ubuntu/damaged_eggs_CNN/api/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now egg-quality
```

Con esto la API queda corriendo en el **puerto 8000** de la IP elástica y sobrevive reinicios.

## 7. Verificar

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true, ...}
```

Desde fuera de AWS (o desde el navegador del celular):

```
http://54.210.100.25:8000/health
```

Si no responde: revisa que el security group tenga el **Custom TCP 8000 abierto** y que `systemctl status egg-quality` esté activo.

## 8. Conectar la app

En `mobile/App.js` (o desde el botón ⚙️ Configurar API de la app) usa:

```
http://54.210.100.25:8000
```

No se necesita HTTPS ni misma red: la IP elástica es pública y el puerto está abierto. Probar con `GET /health` desde el celular es la comprobación más rápida.

## 9. Costos y apagado

- t3.medium ≈ USD 0.04/hora. **Detén la instancia** al terminar la demo ( Elastic IP libre mientras la instancia está apagada no cobra si está asociada).
- Para la sustentación: encender instancia → la IP no cambia → app funciona igual.

## 10. Contrato de preprocesamiento (debe coincidir)

| Parámetro | Valor |
|---|---|
| Tamaño | 100x100 |
| Canales | RGB |
| Normalización | /255 |
| Formato API | multipart/form-data, campo `file` (jpg/png/webp) |

Regla crítica: el preprocesamiento de la API debe ser **idéntico** al de entrenamiento. Si la imagen llega a otra resolución o sin normalizar, la predicción no vale. La app móvil muestra instrucciones de encuadre para que el huevo llegue bien centrado (el modelo fue entrenado con crops de bbox).
