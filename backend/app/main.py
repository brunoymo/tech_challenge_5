import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="API de Previsão de Defasagem Escolar - Passos Mágicos",
    description="API para prever o risco de defasagem de alunos usando Machine Learning.",
    version="1.0.0"
)

# CORS — permite que o frontend React se comunique com a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Caminhos de artefatos
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(__file__)
MODEL_DIR      = os.path.join(BASE_DIR, '../model')
DATA_DIR       = os.path.join(BASE_DIR, '../data/raw')
MODEL_PATH     = os.path.join(MODEL_DIR, 'modelo_defasagem.pkl')
COLS_PATH      = os.path.join(MODEL_DIR, 'colunas.pkl')
METRICAS_PATH  = os.path.join(MODEL_DIR, 'metricas.json')
FEAT_IMP_PATH  = os.path.join(MODEL_DIR, 'feature_importance.json')

# Carregar modelo na inicialização (se existir)
modelo  = None
colunas = None
if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
    modelo  = joblib.load(MODEL_PATH)
    colunas = joblib.load(COLS_PATH)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AlunoInput(BaseModel):
    Idade: int
    Fase: int
    IAA:  float
    IEG:  float
    IPS:  float
    IDA:  float
    IPV:  float
    IAN:  float
    INDE: float

class PrevisaoOutput(BaseModel):
    risco_defasagem: int
    probabilidade:   float

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ler_csv_robusto(caminho: str) -> pd.DataFrame:
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(caminho, sep=';', encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não consegui ler {caminho}")


def _converter_float(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(series.str.replace(',', '.', regex=False), errors='coerce')
    return pd.to_numeric(series, errors='coerce')


_VALORES_ATIVOS = {'cursando', 'ativo', 'sim'}


def _filtrar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    """Remove alunos inativos via coluna 'Ativo/ Inativo' (2024). Mantém tudo se ausente."""
    for col in ('Ativo/ Inativo', 'Ativo/ Inativo.1'):
        if col in df.columns:
            mask = df[col].astype(str).str.strip().str.lower().isin(_VALORES_ATIVOS)
            return df[mask].copy()
    return df


def _carregar_analytics() -> pd.DataFrame:
    """Carrega os três CSVs, filtra alunos ativos e retorna DataFrame unificado."""
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
        # Remove colunas duplicadas geradas pelo rename (ex: INDE 2023 e INDE 22 -> INDE)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        if ano == 2024:
            df = _filtrar_ativos(df)
        df['Ano'] = ano
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Normalizar colunas numéricas
    for col in ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'Defasagem']:
        if col in df.columns:
            df[col] = _converter_float(df[col])

    # Idade
    if 'Idade' in df.columns:
        df['Idade'] = _converter_float(df['Idade'])

    return df


# ---------------------------------------------------------------------------
# Endpoints principais
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Previsão de Defasagem Escolar - Passos Mágicos"}


@app.get("/health")
def health_check():
    """Health-check para monitoramento e Docker HEALTHCHECK."""
    model_loaded = modelo is not None and colunas is not None
    return {
        "status":       "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version":      app.version,
    }


@app.get("/debug")
def debug_paths():
    """Debug: expõe caminhos e existência de arquivos dentro do Docker."""
    csv_files = [
        os.path.join(DATA_DIR, 'dataset_2022.csv'),
        os.path.join(DATA_DIR, 'dataset_2023.csv'),
        os.path.join(DATA_DIR, 'dataset_2024.csv'),
    ]
    return {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "cwd": os.getcwd(),
        "csv_exists": {f: os.path.exists(f) for f in csv_files},
    }



def predict(aluno: AlunoInput):
    if modelo is None or colunas is None:
        raise HTTPException(status_code=503, detail="Modelo não encontrado. Execute train.py primeiro.")

    dados_df = pd.DataFrame([aluno.model_dump()])
    dados_df  = dados_df[colunas]

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
    """Retorna as métricas do último treinamento (accuracy, ROC-AUC, F1, etc.)."""
    if not os.path.exists(METRICAS_PATH):
        raise HTTPException(status_code=404, detail="Métricas não encontradas. Execute train.py primeiro.")
    with open(METRICAS_PATH) as f:
        return json.load(f)


@app.get("/feature-importance")
def get_feature_importance():
    """Retorna a importância de cada feature no modelo Random Forest."""
    if not os.path.exists(FEAT_IMP_PATH):
        raise HTTPException(status_code=404, detail="Feature importances não encontradas. Execute train.py.")
    with open(FEAT_IMP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Endpoints de analytics (dados reais dos CSVs)
# ---------------------------------------------------------------------------
@app.get("/analytics/stats")
def get_stats():
    """
    Estatísticas gerais: total de alunos, INDE médio por ano,
    distribuição por pedra e contagem de alunos em risco.
    """
    try:
        df = _carregar_analytics()
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Erro ao carregar dados: {str(e)}\n{traceback.format_exc()}")

    df_validos = df.dropna(subset=['Defasagem'])
    df_validos = df_validos[pd.to_numeric(df_validos['Defasagem'], errors='coerce').notna()]
    df_validos['Defasagem'] = df_validos['Defasagem'].astype(float).astype(int)

    total_alunos = int(df_validos['RA'].nunique()) if 'RA' in df_validos.columns else int(len(df_validos))
    em_risco     = int((df_validos['Defasagem'] < 0).sum())
    inde_medio   = round(float(df_validos['INDE'].mean()), 2) if 'INDE' in df_validos.columns else None

    # INDE médio por ano
    inde_por_ano = []
    if 'INDE' in df_validos.columns:
        for ano, grp in df_validos.groupby('Ano'):
            inde_por_ano.append({"ano": int(ano), "INDE": round(float(grp['INDE'].mean()), 3)})

    # Distribuição por pedra (último pedra disponível)
    dist_pedras = []
    if 'Pedra' in df_validos.columns:
        for pedra, grp in df_validos.groupby('Pedra'):
            if str(pedra).strip():
                dist_pedras.append({"pedra": str(pedra), "quantidade": int(len(grp))})
        dist_pedras.sort(key=lambda x: x["quantidade"], reverse=True)

    return {
        "total_alunos":  total_alunos,
        "em_risco":      em_risco,
        "pct_risco":     round(em_risco / total_alunos * 100, 1) if total_alunos else 0,
        "inde_medio":    inde_medio,
        "inde_por_ano":  inde_por_ano,
        "dist_pedras":   dist_pedras,
    }


@app.get("/analytics/evolucao")
def get_evolucao():
    """
    Série temporal dos indicadores médios por ano:
    INDE, IAA, IEG, IPS, IDA, IPV, IAN.
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
    """Distribuição de risco de defasagem por fase/ano."""
    df = _carregar_analytics()
    df = df.dropna(subset=['Defasagem', 'Fase'])
    df['Defasagem'] = df['Defasagem'].astype(float).astype(int)
    df['RISCO'] = (df['Defasagem'] < 0).astype(int)

    # Normalizar Fase
    def _parse_fase(v):
        s = str(v).strip().upper()
        if s == 'ALFA':
            return 0
        import re
        m = re.search(r'\d+', s)
        return int(m.group()) if m else None

    df['Fase_num'] = df['Fase'].apply(_parse_fase)
    df = df.dropna(subset=['Fase_num'])
    df['Fase_num'] = df['Fase_num'].astype(int)

    resultado = []
    for fase, grp in df.groupby('Fase_num'):
        resultado.append({
            "fase":       int(fase),
            "total":      int(len(grp)),
            "em_risco":   int(grp['RISCO'].sum()),
            "pct_risco":  round(float(grp['RISCO'].mean()) * 100, 1),
        })

    return sorted(resultado, key=lambda x: x["fase"])
