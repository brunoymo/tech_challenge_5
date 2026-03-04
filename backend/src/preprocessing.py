import pandas as pd
import numpy as np

# Mapeamento de colunas para cada ano
_RENAME_2022 = {
    'Idade 22': 'Idade',
    'INDE 22': 'INDE',
    'Defas': 'Defasagem',
    'Pedra 22': 'Pedra',
}
_RENAME_2023 = {
    'INDE 2023': 'INDE',
    'Pedra 2023': 'Pedra',
    # Idade e Defasagem já têm o nome correto em 2023/2024
}
_RENAME_2024 = {
    'INDE 2024': 'INDE',
    'Pedra 2024': 'Pedra',
}

# Colunas usadas no modelo preditivo
COLUNAS_MODELO = ['Idade', 'Fase', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE']

# Colunas numéricas com decimal vírgula
COLS_FLOAT = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE']


def _ler_csv(caminho: str) -> pd.DataFrame:
    """Lê CSV com fallback de encoding."""
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(caminho, sep=';', encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não foi possível ler {caminho}")


# Valores aceitos como aluno ativo (case-insensitive)
_VALORES_ATIVOS = {'cursando', 'ativo', 'sim'}


def _filtrar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove alunos inativos usando a coluna 'Ativo/ Inativo' (presente apenas no
    dataset 2024). Se a coluna não existir, todos os registros são mantidos.
    Pandas renomeia duplicatas para 'Ativo/ Inativo.1' — ambas são consideradas.
    """
    for col in ('Ativo/ Inativo', 'Ativo/ Inativo.1'):
        if col in df.columns:
            mask = df[col].astype(str).str.strip().str.lower().isin(_VALORES_ATIVOS)
            antes = len(df)
            df = df[mask].copy()
            removidos = antes - len(df)
            if removidos:
                print(f"  [filtro ativo] Removidos {removidos} alunos inativos (coluna '{col}')")
            break  # usar apenas a primeira coluna encontrada
    return df


def carregar_dados(caminho_2022: str, caminho_2023: str, caminho_2024: str) -> pd.DataFrame:
    """Carrega e concatena os três datasets, normalizando colunas e filtrando ativos."""
    df_2022 = _ler_csv(caminho_2022).rename(columns=_RENAME_2022)
    df_2023 = _ler_csv(caminho_2023).rename(columns=_RENAME_2023)
    df_2024 = _filtrar_ativos(_ler_csv(caminho_2024).rename(columns=_RENAME_2024))

    df_2022['Ano'] = 2022
    df_2023['Ano'] = 2023
    df_2024['Ano'] = 2024

    colunas_interesse = COLUNAS_MODELO + ['Defasagem', 'Pedra', 'Ano', 'Gênero', 'Instituição de ensino']

    def _filtrar(df):
        return df[[c for c in colunas_interesse if c in df.columns]]

    df_final = pd.concat([_filtrar(df_2022), _filtrar(df_2023), _filtrar(df_2024)], ignore_index=True)
    return df_final


def _converter_float(series: pd.Series) -> pd.Series:
    """Converte série string para float, tratando vírgula como decimal."""
    if series.dtype == object:
        return pd.to_numeric(series.str.replace(',', '.', regex=False), errors='coerce')
    return pd.to_numeric(series, errors='coerce')


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # Converter colunas numéricas
    for col in COLS_FLOAT:
        if col in df_clean.columns:
            df_clean[col] = _converter_float(df_clean[col])
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Idade
    if 'Idade' in df_clean.columns:
        df_clean['Idade'] = _converter_float(df_clean['Idade'])
        df_clean['Idade'] = df_clean['Idade'].fillna(df_clean['Idade'].median()).astype(int)

    # Defasagem
    if 'Defasagem' in df_clean.columns:
        df_clean['Defasagem'] = _converter_float(df_clean['Defasagem'])
        df_clean = df_clean.dropna(subset=['Defasagem'])
        df_clean['Defasagem'] = df_clean['Defasagem'].astype(int)

    # Fase: ALFA → 0, números → int
    if 'Fase' in df_clean.columns:
        def _parse_fase(v):
            s = str(v).strip().upper()
            if s == 'ALFA':
                return 0
            extracted = pd.Series([s]).str.extract(r'(\d+)')[0].iloc[0]
            return int(extracted) if extracted else 0
        df_clean['Fase'] = df_clean['Fase'].apply(_parse_fase)

    return df_clean


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variável alvo binária RISCO (1 = defasado, 0 = no prazo)."""
    df_f = df.copy()
    if 'Defasagem' in df_f.columns:
        df_f['RISCO'] = (df_f['Defasagem'] < 0).astype(int)
        df_f = df_f.drop('Defasagem', axis=1)
    return df_f


def preprocessar_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = limpar_dados(df)
    df = engenharia_features(df)
    return df
