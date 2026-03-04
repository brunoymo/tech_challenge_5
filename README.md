# Datathon Passos Mágicos - Previsão de Defasagem Escolar

Este projeto é uma solução completa de Machine Learning para prever o risco de defasagem escolar de estudantes da Associação Passos Mágicos, conforme os requisitos do Datathon FIAP — Fase 5.

## 1. Visão Geral do Projeto
**Objetivo:** Desenvolver um modelo preditivo capaz de estimar o risco de defasagem escolar de cada estudante, utilizando dados de 2022, 2023 e 2024.

**Solução Proposta:** Pipeline completo de Machine Learning (MLOps) contendo pré-processamento, treinamento de modelo, avaliação detalhada e deploy via API REST (FastAPI) empacotada em Docker com dashboard React.

**Stack Tecnológica:**
- **ML:** scikit-learn (RandomForestClassifier), pandas, numpy
- **API:** FastAPI + Uvicorn
- **Frontend:** React + Vite + TypeScript + Recharts
- **Testes:** pytest
- **Empacotamento:** Docker + Docker Compose

## 2. Estrutura do Projeto
```
projeto/
├── backend/
│   ├── app/
│   │   └── main.py              # API FastAPI (predict, analytics, metrics)
│   ├── src/
│   │   ├── preprocessing.py     # Pipeline de limpeza e engenharia de features
│   │   ├── train.py             # Treinamento do modelo + salva artefatos
│   │   └── evaluate.py          # Avaliação detalhada (métricas, ROC, feature importance)
│   ├── tests/
│   │   └── test_api.py          # Testes unitários da API
│   ├── model/                   # Artefatos do modelo (.pkl, .json)
│   │   ├── modelo_defasagem.pkl
│   │   ├── colunas.pkl
│   │   ├── metricas.json
│   │   └── feature_importance.json
│   ├── data/raw/                # Datasets (2022, 2023, 2024)
│   ├── Dockerfile
│   └── requirements.txt
├── src/
│   └── components/
│       ├── Analytics.tsx        # Dashboard com dados reais da API
│       └── Predictor.tsx        # Simulador de risco
├── eda_passos_magicos.ipynb     # Notebook de Análise Exploratória
└── docker-compose.yml
```

## 3. Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Mensagem de boas-vindas |
| GET | `/health` | Status da API e do modelo |
| POST | `/predict` | Predição de risco para um aluno |
| GET | `/metrics` | Métricas do último treinamento (ROC-AUC, F1, etc.) |
| GET | `/feature-importance` | Importância das features do modelo |
| GET | `/analytics/stats` | Estatísticas gerais (totais, INDE médio, pedras) |
| GET | `/analytics/evolucao` | Evolução dos indicadores por ano |
| GET | `/analytics/risco-por-fase` | Risco de defasagem por fase |

## 4. Instruções de Deploy

### Pré-requisito: Treinar o modelo
```bash
cd backend/src
python train.py
```
Gera `model/modelo_defasagem.pkl`, `model/colunas.pkl`, `model/metricas.json` e `model/feature_importance.json`.

### Rodando com Docker Compose (recomendado)
```bash
docker-compose up --build
```
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:3000`

### Rodando localmente (sem Docker)
```bash
# Backend
cd backend
pip install -r requirements.txt
python src/train.py          # treinar (só na primeira vez)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (outro terminal)
npm install
npm run dev
```

## 5. Análise Exploratória
Abra `eda_passos_magicos.ipynb` para visualizar:
- Distribuição do INDE por ano
- Evolução dos indicadores (IAA, IEG, IPS, IDA, IPV, IAN)
- Distribuição por Pedra (Quartzo, Ágata, Ametista, Topázio)
- Análise de risco por ano e fase
- Matriz de correlação
- Feature importance e curva ROC do modelo

## 6. Exemplo de Chamada à API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Idade":14,"Fase":3,"IAA":7.0,"IEG":7.5,"IPS":6.5,"IDA":6.0,"IPV":8.0,"IAN":6.0,"INDE":7.0}'
```
**Resposta:**
```json
{"risco_defasagem": 0, "probabilidade": 0.12}
```


## 1. Visão Geral do Projeto
**Objetivo:** Desenvolver um modelo preditivo capaz de estimar o risco de defasagem escolar de cada estudante, utilizando dados de 2022, 2023 e 2024.

**Solução Proposta:** Construção de uma pipeline de Machine Learning (MLOps) contendo pré-processamento, treinamento de modelo, e deploy via API REST (FastAPI) empacotada em Docker.

**Stack Tecnológica:**
- **Linguagem:** Python 3.10
- **Frameworks de ML:** scikit-learn, pandas, numpy
- **API:** FastAPI
- **Serialização:** joblib
- **Testes:** pytest
- **Empacotamento:** Docker

## 2. Estrutura do Projeto
```bash
backend/
├── app/
│   └── main.py              # Arquivo principal da API (FastAPI)
├── src/
│   ├── preprocessing.py     # Funções de limpeza e engenharia de features
│   ├── train.py             # Script de treinamento do modelo
│   └── evaluate.py          # Script de avaliação e métricas (A ser implementado)
├── tests/
│   └── test_api.py          # Testes unitários da API
├── model/                   # Modelos serializados (.pkl)
├── data/                    # Datasets brutos e processados
├── Dockerfile               # Dockerfile para empacotamento
└── requirements.txt         # Dependências do projeto
```

## 3. Instruções de Deploy (Como subir o ambiente)

### Pré-requisitos
- Docker instalado
- Python 3.10+ (para rodar localmente sem Docker)

### Rodando com Docker Compose (recomendado)
Sobe backend + frontend com um único comando:
```bash
# Na raiz do projeto
docker-compose up --build
```
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:3000`

### Rodando apenas o backend com Docker
1. Construa a imagem Docker:
   ```bash
   cd backend
   docker build -t passos-magicos-api .
   ```
2. Execute o contêiner:
   ```bash
   docker run -d -p 8000:8000 passos-magicos-api
   ```
3. Acesse a documentação interativa da API (Swagger) em: `http://localhost:8000/docs`

### Rodando Localmente (Sem Docker)
1. Instale as dependências:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Treine o modelo (Gera o arquivo `.pkl`):
   ```bash
   python src/train.py
   ```
3. Inicie a API:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 4. Exemplos de Chamadas à API
Você pode testar a API via cURL:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Idade": 12,
  "Fase": 2,
  "IAA": 7.0,
  "IEG": 7.5,
  "IPS": 6.5,
  "IDA": 6.0,
  "IPV": 8.0,
  "IAN": 6.0,
  "INDE": 7.0
}'
```
**Output Esperado:**
```json
{
  "risco_defasagem": 0,
  "probabilidade": 0.15
}
```
