"""
Pruebas del endpoint POST /predict del Egg Quality API.

Cubre:
- Imagen válida (PNG generado en memoria) -> 200 con label/confidence/probabilities.
- Bytes basura (formato inválido) -> 415 (tipo de medio) o 400 (procesamiento).
- Sin campo file -> 422 por validación de FastAPI.
- Modelo no cargado -> 503.
"""
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Agregar el directorio api (padre de tests/) al path para importar main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app  # noqa: E402


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Sustituye el modelo real por un doble pequeño (sin entrenar)."""
    import main as main_module

    class FakeModel:
        def predict(self, array, verbose=0):
            import numpy as np
            return np.array([[0.9, 0.1]])  # clase 0 (good) con confianza 0.9

    monkeypatch.setattr(main_module, 'model', FakeModel())


def make_png_bytes(size=(64, 64)):
    buffer = io.BytesIO()
    Image.new('RGB', size, color=(200, 180, 120)).save(buffer, format='PNG')
    return buffer.getvalue()


def test_predict_imagen_valida():
    client = TestClient(app)
    response = client.post(
        '/predict',
        files={'file': ('huevo.png', make_png_bytes(), 'image/png')},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert 'label' in data and data['label'] in {'good', 'crack'}
    assert 0.0 <= data['confidence'] <= 1.0
    assert set(data['probabilities']) == {'good', 'crack'}


def test_predict_formato_invalido():
    client = TestClient(app)
    junk = b'\x00\x01\x02 not an image at all \xff\xfe'
    response = client.post(
        '/predict',
        files={'file': ('basura.jpg', junk, 'image/jpeg')},
    )
    assert response.status_code in {400, 415}, response.text


def test_predict_campo_file_faltante():
    client = TestClient(app)
    response = client.post('/predict', data={})
    assert response.status_code == 422, response.text


def test_predict_sin_modelo():
    import main as main_module

    main_module.model = None
    client = TestClient(app)
    response = client.post(
        '/predict',
        files={'file': ('huevo.png', make_png_bytes(), 'image/png')},
    )
    assert response.status_code == 503, response.text