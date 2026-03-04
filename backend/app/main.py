"""
main.py
=======
API REST do sistema de previsão de defasagem escolar — Passos Mágicos.
Construída com FastAPI, empacotada em Docker e deployada no Render.

Endpoints disponíveis:
  GET  /               → Mensagem de boas-vindas
  GET  /health         → Status da API e do modelo (usado pelo frontend)
  POST /predict        → Predição de risco para um aluno
  GET  /metrics        → Métricas do modelo treinado (ROC-AUC, F1, etc.)
  GET  /feature-importance    → Importância de cada feature no Random Forest
  GET  /analytics/stats       → Estatísticas gerais dos dados (KPIs)
  GET  /analytics/evolucao    → Evolução dos indicadores por ano
  GET  /analytics/risco-por-fase → Distribuição de risco por fase escolar

Uso local:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Inicialização da aplicação FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Previsão de Defasagem Escolar — Passos Mágicos",
    description=(
        "API REST para prever risco de defasagem escolar usando um "
        "Random Forest treinado com dados de 2022, 2023 e 2024. "
        "Também fornece endpoints de analytics baseados nos CSVs originais."
    ),
    version="1.0.0",
)

# Middleware CORS — permite chamadas do frontend React hospedado em outro domínio.
# Em produção, substituir allow_origins=["*"] pelo domínio específico do dashboard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Aceita requisições de qualquer origem
    allow_credentials=True,
    allow_methods=["*"],       # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Caminhos dos artefatos gerados pelo train.py
# Os caminhos são relativos ao diretório deste arquivo (app/).
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(__file__)
MODEL_DIR     = os.path.join(BASE_DIR, '../model')
DATA_DIR      = os.path.join(BASE_DIR, '../data/raw')
MODEL_PATH    = os.path.join(MODEL_DIR, 'modelo_defasagem.pkl')
COLS_PATH     = os.path.join(MODEL_DIR, 'colunas.pkl')
METRICAS_PATH = os.path.join(MODEL_DIR, 'metricas.json')
FEAT_IMP_PATH = os.path.join(MODEL_DIR, 'feature_importance.json')

# ---------------------------------------------------------------------------
# Carregamento do modelo na inicialização do servidor
# O modelo é carregado uma única vez ao subir a API (não a cada requisição).
# Se os arquivos não existirem (ex: ambiente sem artefatos), a API funciona
# em modo degradado — endpoints de analytics ainda respondem, mas /predict
# retorna erro 503.
# ---------------------------------------------------------------------------
modelo  = None
colunas = None
if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
    modelo  = joblib.load(MODEL_PATH)
    colunas = joblib.load(COLS_PATH)
    print(f"[startup] Modelo carregado: {MODEL_PATH}")
else:
    print("[startup] AVISO: Modelo não encontrado. Execute backend/src/train.py primeiro.")

# ---------------------------------------------------------------------------
# Schemas de entrada e saída (Pydantic)
# ---------------------------------------------------------------------------
class AlunoInput(BaseModel):
    """Dados de entrada para predição de risco de um aluno."""
    Idade: int    # Idade em anos
    Fase:  int    # Fase escolar (0=ALFA, 1–8)
    IAA:   float  # Índice de Autoavaliação
    IEG:   float  # Índice de Engajamento
    IPS:   float  # Índice Psicossocial
    IDA:   float  # Índice de Desenvolvimento do Aprendizado
    IPV:   float  # Índice do Ponto de Virada
    IAN:   float  # Índice de Adequação ao Nível
    INDE:  float  # Índice de Desenvolvimento Educacional (composto)

class PrevisaoOutput(BaseModel):
    """Resposta da predição de risco."""
    risco_defasagem: int    # 0 = sem risco, 1 = em risco
    probabilidade:   float  # Probabilidade de risco (0.0 a 1.0)


# ---------------------------------------------------------------------------
# Helpers internos (não expostos como endpoints)
# ---------------------------------------------------------------------------
def _ler_csv_robusto(caminho: str) -> pd.DataFrame:
    """
    Lê um CSV com sep=';' tentando múltiplos encodings (utf-8, latin-1, cp1252).

    Os datasets da Passos Mágicos foram gerados com encodings variados.
    Esta função garante leitura correta em qualquer cenário.
    """
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(caminho, sep=';', encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não consegui ler {caminho} — encoding desconhecido.")


def _converter_float(series: pd.Series) -> pd.Series:
    """Converte Series string para float, substituindo vírgula por ponto."""
    if series.dtype == object:
        return pd.to_numeric(series.str.replace(',', '.', regex=False), errors='coerce')
    return pd.to_numeric(series, errors='coerce')


# Status de ativo aceitos na coluna 'Ativo/ Inativo' (dataset 2024)
_VALORES_ATIVOS = {'cursando', 'ativo', 'sim'}


def _filtrar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove alunos inativos usando a coluna 'Ativo/ Inativo' (exclusiva do dataset 2024).

    Se a coluna não existir (datasets 2022 e 2023), retorna o DataFrame intacto.
    O pandas pode criar variantes com sufixo '.1' por duplicatas — ambas são verificadas.
    """
    for col in ('Ativo/ Inativo', 'Ativo/ Inativo.1'):
        if col in df.columns:
            mask = df[col].astype(str).str.strip().str.lower().isin(_VALORES_ATIVOS)
            return df[mask].copy()
    return df


def _carregar_analytics() -> pd.DataFrame:
    """
    Carrega e unifica os três CSVs para uso nos endpoints de analytics.

    Realiza as mesmas normalizações do pipeline de treinamento:
    - Renomeia colunas inconsistentes entre anos.
    - Remove colunas duplicadas geradas pelo rename.
    - Filtra alunos inativos do dataset 2024.
    - Converte colunas numéricas (vírgula → ponto).

    Returns:
        DataFrame unificado com colunas normalizadas.
    """
    # Mapa de renomeação unificado para os três anos
    rename_map = {
        'Idade 22': 'Idade', 'INDE 22': 'INDE',
        'Defas': 'Defasagem', 'Pedra 22': 'Pedra',
        'INDE 2023': 'INDE', 'Pedra 2023': 'Pedra',
        'INDE 2024': 'INDE', 'Pedra 2024': 'Pedra',
    }

    frames = []
    for ano, path in [
        (2022, os.path.join(DATA_DIR, 'dataset_2022.csv')),
        (2023, os.path.join(DATA_DIR, 'dataset_2023.csv')),
        (2024, os.path.join(DATA_DIR, 'dataset_2024.csv')),
    ]:
        df = _ler_csv_robusto(path).rename(columns=rename_map)

        # Remove colunas duplicadas geradas quando dois campos distintos
        # recebem o mesmo nome após o rename (ex: 'INDE 2023' e 'INDE 22' → 'INDE')
        df = df.loc[:, ~df.columns.duplicated()].copy()

        if ano == 2024:
            df = _filtrar_ativos(df)

        df['Ano'] = ano
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Normaliza colunas numéricas (vírgula decimal → ponto decimal)
    for col in ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'Defasagem', 'Idade']:
        if col in df.columns:
            df[col] = _converter_float(df[col])

    return df


# ---------------------------------------------------------------------------
# Endpoints principais
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    """Endpoint raiz — confirma que a API está no ar."""
    return {"message": "Bem-vindo à API de Previsão de Defasagem Escolar — Passos Mágicos"}


@app.get("/health")
def health_check():
    """
    Health-check para monitoramento externo e para o dashboard frontend.

    Responde com status 'ok' se o modelo estiver carregado, ou 'degraded'
    caso os artefatos não tenham sido encontrados no startup.
    O frontend usa este endpoint para exibir o badge 'API Online/Offline'.
    """
    model_loaded = modelo is not None and colunas is not None
    return {
        "status":       "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version":      app.version,
    }


@app.post("/predict", response_model=PrevisaoOutput)
def predict(aluno: AlunoInput):
    """
    Prediz o risco de defasagem escolar para um aluno.

    Recebe os indicadores do aluno, aplica o modelo Random Forest carregado
    no startup e retorna a classe predita (0/1) e a probabilidade de risco.

    Returns:
        PrevisaoOutput com risco_defasagem (0=sem risco, 1=em risco)
        e probabilidade (float entre 0.0 e 1.0).

    Raises:
        HTTPException 503 se o modelo não estiver carregado.
        HTTPException 500 em caso de erro durante a inferência.
    """
    if modelo is None or colunas is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não encontrado. Execute backend/src/train.py para gerar os artefatos."
        )

    # Converte o input para DataFrame e seleciona colunas na ordem do treinamento
    dados_df = pd.DataFrame([aluno.model_dump()])
    dados_df = dados_df[colunas]

    try:
        predicao      = modelo.predict(dados_df)[0]
        probabilidade = modelo.predict_proba(dados_df)[0][1]
        return {"risco_defasagem": int(predicao), "probabilidade": float(probabilidade)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


# ---------------------------------------------------------------------------
# Endpoints de métricas e explicabilidade
# ---------------------------------------------------------------------------
@app.get("/metrics")
def get_metrics():
    """
    Retorna as métricas do último treinamento do modelo.

    Lê o arquivo metricas.json gerado por train.py, que contém:
    accuracy, roc_auc, f1, precision, recall, confusion_matrix e roc_curve.
    Utilizado pelo componente ModelMetrics.tsx do dashboard.
    """
    if not os.path.exists(METRICAS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Métricas não encontradas. Execute backend/src/train.py primeiro."
        )
    with open(METRICAS_PATH) as f:
        return json.load(f)


@app.get("/feature-importance")
def get_feature_importance():
    """
    Retorna a importância de cada feature no modelo Random Forest.

    Lê feature_importance.json gerado por train.py.
    Lista ordenada de {feature, importance} em ordem decrescente.
    Utilizado pelo componente ModelMetrics.tsx do dashboard.
    """
    if not os.path.exists(FEAT_IMP_PATH):
        raise HTTPException(
            status_code=404,
            detail="Feature importances não encontradas. Execute backend/src/train.py."
        )
    with open(FEAT_IMP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Endpoints de analytics (leitura direta dos CSVs originais)
# ---------------------------------------------------------------------------
@app.get("/analytics/stats")
def get_stats():
    """
    Estatísticas gerais dos dados reais: KPIs do dashboard Analytics.

    Retorna:
    - total_alunos: número de alunos únicos (ou registros, se sem RA)
    - em_risco / pct_risco: contagem e % de alunos com Defasagem < 0
    - inde_medio: INDE médio geral
    - inde_por_ano: lista de {ano, INDE} com médias por ano
    - dist_pedras: distribuição de alunos por classificação (Quartzo, Ágata, etc.)
    """
    df = _carregar_analytics()

    # Considera apenas registros com Defasagem válida para calcular risco
    df_validos = df.dropna(subset=['Defasagem'])
    df_validos = df_validos[pd.to_numeric(df_validos['Defasagem'], errors='coerce').notna()]
    df_validos['Defasagem'] = df_validos['Defasagem'].astype(float).astype(int)

    # KPIs principais
    total_alunos = int(df_validos['RA'].nunique()) if 'RA' in df_validos.columns else int(len(df_validos))
    em_risco     = int((df_validos['Defasagem'] < 0).sum())
    inde_medio   = round(float(df_validos['INDE'].mean()), 2) if 'INDE' in df_validos.columns else None

    # INDE médio por ano — usado no gráfico de linha do dashboard
    inde_por_ano = []
    if 'INDE' in df_validos.columns:
        for ano, grp in df_validos.groupby('Ano'):
            inde_por_ano.append({"ano": int(ano), "INDE": round(float(grp['INDE'].mean()), 3)})

    # Distribuição por pedra — usado no gráfico de barras/pizza do dashboard
    dist_pedras = []
    if 'Pedra' in df_validos.columns:
        for pedra, grp in df_validos.groupby('Pedra'):
            if str(pedra).strip():  # Ignora valores vazios
                dist_pedras.append({"pedra": str(pedra), "quantidade": int(len(grp))})
        dist_pedras.sort(key=lambda x: x["quantidade"], reverse=True)

    return {
        "total_alunos": total_alunos,
        "em_risco":     em_risco,
        "pct_risco":    round(em_risco / total_alunos * 100, 1) if total_alunos else 0,
        "inde_medio":   inde_medio,
        "inde_por_ano": inde_por_ano,
        "dist_pedras":  dist_pedras,
    }


@app.get("/analytics/evolucao")
def get_evolucao():
    """
    Série temporal dos indicadores médios por ano.

    Retorna lista de {ano, INDE, IAA, IEG, IPS, IDA, IPV, IAN} com os valores
    médios de cada indicador em 2022, 2023 e 2024.
    Utilizado pelo gráfico de linhas 'Evolução dos Indicadores' do dashboard.
    """
    df = _carregar_analytics()
    indicadores = [c for c in ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN'] if c in df.columns]

    resultado = []
    for ano, grp in df.groupby('Ano'):
        row: dict = {"ano": int(ano)}
        for ind in indicadores:
            val = grp[ind].dropna()
            row[ind] = round(float(val.mean()), 3) if len(val) else None
        resultado.append(row)

    return sorted(resultado, key=lambda x: x["ano"])


@app.get("/analytics/risco-por-fase")
def get_risco_por_fase():
    """
    Distribuição de risco de defasagem por fase escolar.

    Normaliza a coluna 'Fase' (strings como 'ALFA' → 0, 'Fase 3' → 3) e
    agrupa os registros, calculando total de alunos, quantidade em risco
    e percentual de risco por fase.
    Utilizado no gráfico de barras 'Risco por Fase' do dashboard.
    """
    import re

    df = _carregar_analytics()
    df = df.dropna(subset=['Defasagem', 'Fase'])
    df['Defasagem'] = df['Defasagem'].astype(float).astype(int)
    df['RISCO'] = (df['Defasagem'] < 0).astype(int)

    def _parse_fase(v):
        """Extrai número ordinal da fase; 'ALFA' → 0; strings inválidas → None."""
        s = str(v).strip().upper()
        if s == 'ALFA':
            return 0
        m = re.search(r'\d+', s)
        return int(m.group()) if m else None

    df['Fase_num'] = df['Fase'].apply(_parse_fase)
    df = df.dropna(subset=['Fase_num'])
    df['Fase_num'] = df['Fase_num'].astype(int)

    resultado = []
    for fase, grp in df.groupby('Fase_num'):
        resultado.append({
            "fase":      int(fase),
            "total":     int(len(grp)),
            "em_risco":  int(grp['RISCO'].sum()),
            "pct_risco": round(float(grp['RISCO'].mean()) * 100, 1),
        })

    return sorted(resultado, key=lambda x: x["fase"])
