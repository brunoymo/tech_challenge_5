"""
preprocessing.py
================
Pipeline de pré-processamento de dados para o modelo de previsão de defasagem
escolar do Datathon Passos Mágicos (FIAP — Fase 5).

Responsabilidades:
  1. Ler os CSVs brutos dos três anos (2022, 2023, 2024) com fallback de encoding.
  2. Renomear colunas inconsistentes entre anos para um esquema unificado.
  3. Filtrar alunos inativos (somente dataset 2024 possui essa coluna).
  4. Converter campos numéricos que usam vírgula como separador decimal.
  5. Normalizar a coluna 'Fase' (string "ALFA" → 0, "Fase 3" → 3, etc.).
  6. Criar a variável alvo binária RISCO (1 = defasado, 0 = no prazo).

Uso típico:
    from preprocessing import carregar_dados, preprocessar_pipeline, COLUNAS_MODELO
    df_bruto = carregar_dados(path_2022, path_2023, path_2024)
    df_proc  = preprocessar_pipeline(df_bruto)
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Mapeamentos de renomeação por ano
# Cada CSV usa nomes ligeiramente diferentes para as mesmas colunas.
# ---------------------------------------------------------------------------
_RENAME_2022 = {
    'Idade 22': 'Idade',    # coluna de idade tem sufixo "22" em 2022
    'INDE 22':  'INDE',     # índice geral com sufixo de ano
    'Defas':    'Defasagem',# nome abreviado no CSV 2022
    'Pedra 22': 'Pedra',    # classificação da pedra com sufixo
}
_RENAME_2023 = {
    'INDE 2023': 'INDE',    # índice geral com sufixo do ano por extenso
    'Pedra 2023': 'Pedra',  # classificação com sufixo
    # Em 2023, Idade e Defasagem já têm o nome canônico.
}
_RENAME_2024 = {
    'INDE 2024': 'INDE',    # mesma lógica para 2024
    'Pedra 2024': 'Pedra',
}

# Features usadas pelo modelo de ML (ordem importa para reprodutibilidade)
COLUNAS_MODELO = ['Idade', 'Fase', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE']

# Colunas numéricas que usam vírgula como separador decimal no CSV
COLS_FLOAT = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE']

# Valores que identificam um aluno como ativo no dataset 2024
_VALORES_ATIVOS = {'cursando', 'ativo', 'sim'}


def _ler_csv(caminho: str) -> pd.DataFrame:
    """
    Lê um CSV com separador ponto-e-vírgula tentando múltiplos encodings.

    Os CSVs da Passos Mágicos foram gerados com diferentes encodings ao longo
    dos anos. A estratégia de fallback garante leitura correta em qualquer caso.

    Args:
        caminho: Caminho absoluto ou relativo para o arquivo .csv.

    Returns:
        DataFrame com todas as colunas lidas como string (dtype=str) para
        evitar inferência incorreta de tipos antes da normalização.

    Raises:
        ValueError: Se nenhum encoding conseguir ler o arquivo.
    """
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(caminho, sep=';', encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não foi possível ler {caminho} — verifique o encoding.")


def _filtrar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove alunos inativos usando a coluna 'Ativo/ Inativo'.

    Presente apenas no dataset de 2024. Quando o pandas encontra dois campos
    com o mesmo nome, renomeia o segundo para 'Ativo/ Inativo.1' — ambas as
    variantes são verificadas para garantir compatibilidade.

    Se nenhuma das colunas existir (datasets 2022 e 2023), retorna o DataFrame
    intacto — todos os registros passam.

    Args:
        df: DataFrame bruto (com colunas já renomeadas).

    Returns:
        DataFrame filtrado com apenas alunos ativos.
    """
    for col in ('Ativo/ Inativo', 'Ativo/ Inativo.1'):
        if col in df.columns:
            # Normaliza para comparação case-insensitive
            mask = df[col].astype(str).str.strip().str.lower().isin(_VALORES_ATIVOS)
            antes = len(df)
            df = df[mask].copy()
            removidos = antes - len(df)
            if removidos:
                print(f"  [filtro ativo] Removidos {removidos} alunos inativos (coluna '{col}')")
            break  # Usar apenas a primeira coluna encontrada
    return df


def carregar_dados(caminho_2022: str, caminho_2023: str, caminho_2024: str) -> pd.DataFrame:
    """
    Carrega e concatena os três datasets anuais em um único DataFrame.

    Fluxo:
      1. Lê cada CSV com encoding robusto.
      2. Renomeia colunas para o esquema unificado.
      3. Filtra alunos inativos do dataset 2024.
      4. Adiciona coluna 'Ano' a cada subset.
      5. Seleciona apenas as colunas de interesse antes de concatenar
         (evita explosão de colunas duplicadas por diferenças de schema).

    Args:
        caminho_2022: Caminho para dataset_2022.csv
        caminho_2023: Caminho para dataset_2023.csv
        caminho_2024: Caminho para dataset_2024.csv

    Returns:
        DataFrame unificado com todas as colunas relevantes.
    """
    # Carrega e renomeia colunas de cada ano
    df_2022 = _ler_csv(caminho_2022).rename(columns=_RENAME_2022)
    df_2023 = _ler_csv(caminho_2023).rename(columns=_RENAME_2023)
    df_2024 = _filtrar_ativos(_ler_csv(caminho_2024).rename(columns=_RENAME_2024))

    # Marca cada registro com o ano de origem
    df_2022['Ano'] = 2022
    df_2023['Ano'] = 2023
    df_2024['Ano'] = 2024

    # Colunas de interesse: features do modelo + metadados úteis para análise
    colunas_interesse = COLUNAS_MODELO + ['Defasagem', 'Pedra', 'Ano', 'Gênero', 'Instituição de ensino']

    def _filtrar(df):
        # Seleciona apenas colunas que existem no DataFrame atual
        return df[[c for c in colunas_interesse if c in df.columns]]

    df_final = pd.concat(
        [_filtrar(df_2022), _filtrar(df_2023), _filtrar(df_2024)],
        ignore_index=True
    )
    return df_final


def _converter_float(series: pd.Series) -> pd.Series:
    """
    Converte uma Series string para float, tratando vírgula como decimal.

    Os CSVs da Passos Mágicos usam vírgula como separador decimal (padrão
    brasileiro). Este helper substitui a vírgula por ponto antes de converter.

    Args:
        series: Series com valores numéricos como strings (ex: "7,5").

    Returns:
        Series float com NaN onde a conversão falhar.
    """
    if series.dtype == object:
        return pd.to_numeric(series.str.replace(',', '.', regex=False), errors='coerce')
    return pd.to_numeric(series, errors='coerce')


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpeza e normalização ao DataFrame consolidado.

    Operações realizadas:
    - Converte colunas numéricas de string para float (tratando vírgula decimal).
    - Preenche valores ausentes com a mediana da coluna.
    - Normaliza a coluna 'Fase': "ALFA" → 0, "Fase 3" → 3, etc.
    - Remove linhas sem valor de Defasagem (instâncias sem rótulo).

    Args:
        df: DataFrame bruto concatenado.

    Returns:
        DataFrame limpo e com tipos corretos.
    """
    df_clean = df.copy()

    # Converter colunas numéricas (vírgula → ponto) e preencher NaN com mediana
    for col in COLS_FLOAT:
        if col in df_clean.columns:
            df_clean[col] = _converter_float(df_clean[col])
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Idade: converter para int após preencher NaN
    if 'Idade' in df_clean.columns:
        df_clean['Idade'] = _converter_float(df_clean['Idade'])
        df_clean['Idade'] = df_clean['Idade'].fillna(df_clean['Idade'].median()).astype(int)

    # Defasagem: converter e remover linhas sem rótulo (não podem ser usadas no treino)
    if 'Defasagem' in df_clean.columns:
        df_clean['Defasagem'] = _converter_float(df_clean['Defasagem'])
        df_clean = df_clean.dropna(subset=['Defasagem'])
        df_clean['Defasagem'] = df_clean['Defasagem'].astype(int)

    # Fase: normalizar valores textuais para inteiro
    if 'Fase' in df_clean.columns:
        def _parse_fase(v):
            """Extrai o número da fase; retorna 0 para 'ALFA'."""
            s = str(v).strip().upper()
            if s == 'ALFA':
                return 0  # Fase ALFA equivale à fase 0
            # Extrai o primeiro número encontrado na string (ex: "Fase 3" → 3)
            extracted = pd.Series([s]).str.extract(r'(\d+)')[0].iloc[0]
            return int(extracted) if extracted else 0
        df_clean['Fase'] = df_clean['Fase'].apply(_parse_fase)

    return df_clean


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável alvo binária RISCO a partir da Defasagem.

    Definição de risco:
        RISCO = 1  →  Defasagem < 0  (aluno está atrasado em relação à fase ideal)
        RISCO = 0  →  Defasagem ≥ 0  (aluno está no prazo ou adiantado)

    A coluna 'Defasagem' é descartada após criar RISCO para evitar vazamento
    de dados (data leakage) no treinamento.

    Args:
        df: DataFrame pós-limpeza com coluna 'Defasagem'.

    Returns:
        DataFrame com coluna RISCO e sem coluna Defasagem.
    """
    df_f = df.copy()
    if 'Defasagem' in df_f.columns:
        df_f['RISCO'] = (df_f['Defasagem'] < 0).astype(int)
        df_f = df_f.drop('Defasagem', axis=1)
    return df_f


def preprocessar_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o pipeline completo de pré-processamento: limpeza + engenharia.

    Atalho conveniente que combina limpar_dados() e engenharia_features()
    em uma única chamada.

    Args:
        df: DataFrame bruto retornado por carregar_dados().

    Returns:
        DataFrame processado, pronto para treinamento ou inferência.
    """
    df = limpar_dados(df)
    df = engenharia_features(df)
    return df
