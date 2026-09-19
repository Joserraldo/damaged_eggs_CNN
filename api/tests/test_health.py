"""
Pruebas unitarias para el endpoint GET /health del Egg Quality API.

Verifica que el endpoint responde 200 y contiene las claves status y model_loaded.
"""
import pytest
from fastapi.testclient import TestClient
import os
import sys

# Agregar el directorio api al path para importar main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Simula que el modelo está cargado para las pruebas de health."""
    # Forzamos que model sea None al inicio, entonces /health debe reportar model_loaded: False
    from main import model as main_model
    monkeypatch.setattr('main.model', None)
    # También forzar MODEL_PATH que no existe
    monkeypatch.setattr('main.MODEL_PATH', None)


def test_health_responds_200():
    """GET /health debe responder 200 y tener las claves status y model_loaded."""
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data, f"Expected 'status' in response, got: {data}"
    assert 'model_loaded' in data, f"Expected 'model_loaded' in response, got: {data}"
    print(f"Health response: {data}")


def test_health_without_model():
    """Cuando no hay modelo, status debe ser 'degraded' y model_loaded debe ser False."""
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['model_loaded'] is False
    assert data['status'] == 'degraded'
    print(f"Health without model: {data}")