# 🎓 Datathon Passos Mágicos — Previsão de Defasagem Escolar

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![React](https://img.shields.io/badge/React-19-blue?logo=react)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![Render](https://img.shields.io/badge/Deploy-Render-purple?logo=render)

> **FIAP — Pós-Tech Machine Learning | Fase 5 | Datathon**  
> Solução completa de MLOps para prever o risco de defasagem escolar de alunos da Associação Passos Mágicos.

---

## 🔗 Links de Produção

| Serviço | URL |
|---------|-----|
| 🖥️ Dashboard | [passos-magicos-dashboard.onrender.com](https://passos-magicos-dashboard.onrender.com) |
| ⚙️ API (Swagger) | [passos-magicos-api-chcj.onrender.com/docs](https://passos-magicos-api-chcj.onrender.com/docs) |

> **Nota:** Os serviços estão hospedados no plano gratuito do Render. Na primeira requisição após inatividade, a API pode demorar ~30 segundos para "acordar".

---

## 1. Visão Geral

**Problema:** A Associação Passos Mágicos acompanha centenas de estudantes em situação de vulnerabilidade. Identificar precocemente quais alunos estão em risco de defasagem escolar é crucial para direcionar recursos de forma eficaz.

**Solução:** Pipeline completo de Machine Learning (MLOps) que:
1. Processa dados históricos de 2022, 2023 e 2024.
2. Treina um **Random Forest** para prever se um aluno apresentará defasagem (RISCO binário).
3. Expõe a predição e analytics via **API REST (FastAPI)**.
4. Visualiza tudo em um **Dashboard React** interativo com gráficos em tempo real.

---

## 2. Arquitetura

```
projeto/
├── backend/                        # Serviço Python/FastAPI
│   ├── app/
│   │   └── main.py                 # API FastAPI — endpoints de predição e analytics
│   ├── src/
│   │   ├── preprocessing.py        # Pipeline de limpeza e engenharia de features
│   │   ├── train.py                # Treinamento do modelo + geração de artefatos
│   │   └── evaluate.py             # Avaliação detalhada (métricas, ROC, feature importance)
│   ├── tests/
│   │   └── test_api.py             # Testes unitários da API com pytest
│   ├── model/                      # Artefatos do modelo (gerados por train.py)
│   │   ├── modelo_defasagem.pkl    # Modelo Random Forest serializado
│   │   ├── colunas.pkl             # Lista de features usadas no treinamento
│   │   ├── metricas.json           # Métricas completas (ROC-AUC, F1, etc.)
│   │   └── feature_importance.json # Importância de cada feature
│   ├── data/raw/                   # Datasets originais (CSV)
│   │   ├── dataset_2022.csv
│   │   ├── dataset_2023.csv
│   │   └── dataset_2024.csv
│   ├── Dockerfile                  # Imagem Docker da API
│   └── requirements.txt
├── src/                            # Frontend React/Vite/TypeScript
│   ├── App.tsx                     # Componente raiz — tabs, health check, sidebar
│   └── components/
│       ├── Analytics.tsx           # 📊 Dashboard com KPIs e gráficos dos dados reais
│       ├── Predictor.tsx           # 🧠 Simulador de risco individual
│       └── ModelMetrics.tsx        # 📈 Métricas do ML (ROC, matriz de confusão, feature importance)
├── eda_passos_magicos.ipynb        # 🔬 Notebook de Análise Exploratória (EDA)
├── render.yaml                     # Blueprint do Render (deploy automatizado)
├── docker-compose.yml              # Orquestração local (API + Frontend)
├── index.html                      # Entry point do frontend
├── vite.config.ts                  # Configuração do Vite
└── package.json
```

---

## 3. Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| **ML** | scikit-learn (RandomForestClassifier), pandas, numpy |
| **API** | FastAPI 0.110 + Uvicorn |
| **Frontend** | React 19 + Vite + TypeScript + Recharts + TailwindCSS |
| **Testes** | pytest |
| **Empacotamento** | Docker + Docker Compose |
| **Deploy** | Render (Docker service + Static Site) |

---

## 4. Features e Metodologia ML

### Variáveis de entrada
| Feature | Descrição |
|---------|-----------|
| `Idade` | Idade do aluno em anos |
| `Fase` | Fase escolar atual (0 = ALFA, 1–8) |
| `IAA` | Índice de Autoavaliação |
| `IEG` | Índice de Engajamento |
| `IPS` | Índice Psicossocial |
| `IDA` | Índice de Desenvolvimento do Aprendizado |
| `IPV` | Índice do Ponto de Virada |
| `IAN` | Índice de Adequação ao Nível |
| `INDE` | Índice de Desenvolvimento Educacional (composto) |

### Variável alvo
- `RISCO = 1` → Aluno com defasagem escolar (Defasagem < 0)
- `RISCO = 0` → Aluno no prazo

### Modelo
- **Algoritmo:** RandomForestClassifier
- **Hiperparâmetros:** 200 estimadores, `class_weight='balanced'` (lida com desbalanceamento), `random_state=42`
- **Split:** 80% treino / 20% teste, estratificado

---

## 5. Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Mensagem de boas-vindas |
| GET | `/health` | Status da API e do modelo |
| POST | `/predict` | Predição de risco para um aluno |
| GET | `/metrics` | Métricas do modelo (ROC-AUC, F1, etc.) |
| GET | `/feature-importance` | Importância das features |
| GET | `/analytics/stats` | Estatísticas gerais (totais, INDE médio, pedras) |
| GET | `/analytics/evolucao` | Evolução dos indicadores por ano |
| GET | `/analytics/risco-por-fase` | Risco de defasagem por fase escolar |

### Exemplo de chamada

```bash
curl -X POST https://passos-magicos-api-chcj.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"Idade":14,"Fase":3,"IAA":7.0,"IEG":7.5,"IPS":6.5,"IDA":6.0,"IPV":8.0,"IAN":6.0,"INDE":7.0}'
```

**Resposta:**
```json
{
  "risco_defasagem": 0,
  "probabilidade": 0.12
}
```

---

## 6. Como Executar

### Pré-requisito: treinar o modelo (apenas na primeira vez)
```bash
cd backend/src
python train.py
```
Gera os artefatos em `backend/model/`.

### Opção A — Docker Compose (recomendado)
```bash
# Na raiz do projeto
docker-compose up --build
```
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:3000`

### Opção B — Localmente (sem Docker)
```bash
# Terminal 1 — API
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
npm install
npm run dev
```

### Opção C — Deploy no Render (Produção)
O projeto inclui `render.yaml` (Blueprint). Para deployar:
1. Faça fork do repositório no GitHub.
2. No Render, clique em **New → Blueprint** e conecte seu repositório.
3. O Render criará automaticamente os dois serviços.

---

## 7. Testes

```bash
cd backend
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## 8. Análise Exploratória

Abra `eda_passos_magicos.ipynb` para ver:
- Distribuição do INDE por ano (boxplots)
- Evolução temporal dos indicadores (2022→2024)
- Distribuição por Pedra (Quartzo, Ágata, Ametista, Topázio)
- Análise de risco de defasagem por Fase escolar
- Heatmap de correlação entre indicadores
- Feature importance e Curva ROC do modelo treinado

```bash
pip install jupyter matplotlib seaborn scikit-learn
jupyter notebook eda_passos_magicos.ipynb
```
