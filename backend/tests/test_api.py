from fastapi.testclient import TestClient
import sys
import os

# Adiciona o diretório app ao path para importar o main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Bem-vindo" in response.json()["message"]

def test_predict_sem_modelo():
    """Verifica que a API aceita o payload correto.
    Retorna 200 com predição se o modelo estiver disponível, ou 503 caso contrário."""
    payload = {
        "Idade": 12,
        "Fase": 2,
        "IAA": 7.0,
        "IEG": 7.5,
        "IPS": 6.5,
        "IDA": 6.0,
        "IPV": 8.0,
        "IAN": 6.0,
        "INDE": 7.0
    }
    response = client.post("/predict", json=payload)
    # 200 se o modelo estiver treinado, 503 se ainda não foi gerado
    assert response.status_code in [200, 503]


def test_predict_payload_invalido():
    """Campos obrigatórios ausentes devem retornar HTTP 422 Unprocessable Entity."""
    response = client.post("/predict", json={"Idade": 12})  # campos incompletos
    assert response.status_code == 422


def test_health():
    """Endpoint /health deve sempre retornar 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
