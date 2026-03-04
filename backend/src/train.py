"""
train.py
========
Script de treinamento do modelo de ML para previsão de defasagem escolar.
Parte do pipeline MLOps do Datathon Passos Mágicos — FIAP Fase 5.

Fluxo de execução:
  1. Carrega e pré-processa os três datasets (2022, 2023, 2024).
  2. Divide em treino/teste (80/20, estratificado).
  3. Treina um RandomForestClassifier com class_weight='balanced'.
  4. Avalia com métricas completas (ROC-AUC, F1, Precision, Recall).
  5. Serializa o modelo e todos os artefatos em backend/model/.

Artefatos gerados:
  - modelo_defasagem.pkl     → Modelo Random Forest serializado com joblib
  - colunas.pkl              → Lista de features na ordem do treinamento
  - metricas.json            → Métricas de avaliação + curva ROC + matriz de confusão
  - feature_importance.json  → Importância relativa de cada feature (ordenado)

Uso:
    cd backend/src
    python train.py
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix, roc_curve,
)
import joblib

# Importa o pipeline de pré-processamento local
from preprocessing import carregar_dados, preprocessar_pipeline, COLUNAS_MODELO

# ---------------------------------------------------------------------------
# Caminhos de arquivos — relativos ao diretório deste script
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, '../data/raw')    # CSVs brutos
MODEL_DIR = os.path.join(BASE_DIR, '../model')        # Artefatos de saída


def treinar_modelo(caminho_modelo_saida: str) -> None:
    """
    Pipeline completo de treinamento: carga → pré-proc → treino → avaliação → salvar.

    Args:
        caminho_modelo_saida: Caminho onde o arquivo .pkl do modelo será salvo.
    """
    # Garante que o diretório de saída existe antes de tentar salvar
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- 1. Carga de dados ----
    print("Carregando dados dos três anos...")
    df_bruto = carregar_dados(
        os.path.join(DATA_DIR, 'dataset_2022.csv'),
        os.path.join(DATA_DIR, 'dataset_2023.csv'),
        os.path.join(DATA_DIR, 'dataset_2024.csv'),
    )

    # ---- 2. Pré-processamento ----
    print("Pré-processando dados (limpeza + engenharia de features)...")
    df_proc = preprocessar_pipeline(df_bruto)

    # Mantém apenas as colunas esperadas pelo modelo + variável alvo
    # (elimina colunas extras como 'Pedra', 'Gênero', 'Ano', etc.)
    col_modelo = [c for c in COLUNAS_MODELO if c in df_proc.columns]
    df_proc = df_proc[col_modelo + ['RISCO']].dropna()

    X = df_proc[col_modelo]  # Features (entrada do modelo)
    y = df_proc['RISCO']     # Variável alvo (0 = sem risco, 1 = em risco)

    print(f"Dataset final: {len(X)} amostras | RISCO=1: {y.sum()} ({y.mean()*100:.1f}%)")

    # ---- 3. Divisão treino/teste ----
    # stratify=y garante proporção de classes semelhante nos dois subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 4. Treinamento ----
    print("Treinando RandomForestClassifier...")
    modelo = RandomForestClassifier(
        n_estimators=200,     # Número de árvores — equilibrio entre custo e variância
        max_depth=None,       # Permite crescimento irrestrito das árvores
        min_samples_leaf=2,   # Evita folhas com uma única amostra (overfitting)
        random_state=42,      # Reprodutibilidade
        class_weight='balanced',  # Compensa o desbalanceamento entre classes
        n_jobs=-1,            # Usa todos os núcleos disponíveis
    )
    modelo.fit(X_train, y_train)

    # ---- 5. Avaliação no conjunto de teste ----
    y_pred = modelo.predict(X_test)          # Classe predita (0 ou 1)
    y_prob = modelo.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva (RISCO=1)

    # Métricas de classificação
    roc_auc = roc_auc_score(y_test, y_prob)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)

    # Curva ROC — usada no dashboard de métricas do frontend
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Imprime relatório completo no console
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")

    # ---- 6. Salvar métricas em JSON ----
    # Estrutura consumida pelo endpoint GET /metrics da API e pelo frontend
    metricas = {
        "accuracy":        round(acc, 4),
        "roc_auc":         round(roc_auc, 4),
        "f1":              round(f1, 4),
        "precision":       round(prec, 4),
        "recall":          round(rec, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            # Arredonda pontos da curva para reduzir tamanho do JSON
            "fpr": [round(v, 4) for v in fpr.tolist()],
            "tpr": [round(v, 4) for v in tpr.tolist()],
        },
        "n_amostras": int(len(X)),         # Total de amostras disponíveis
        "n_treino":   int(len(X_train)),    # Amostras usadas no treino (80%)
        "n_teste":    int(len(X_test)),     # Amostras usadas no teste (20%)
    }
    metricas_path = os.path.join(MODEL_DIR, 'metricas.json')
    with open(metricas_path, 'w') as f:
        json.dump(metricas, f, indent=2)
    print(f"Métricas salvas em {metricas_path}")

    # ---- 7. Salvar feature importances em JSON ----
    # Importâncias retornadas pelo RandomForest (média de ganho de impureza)
    importances = modelo.feature_importances_
    fi = sorted(
        [{"feature": col, "importance": round(float(imp), 4)}
         for col, imp in zip(col_modelo, importances)],
        key=lambda x: x["importance"],   # Ordena do mais ao menos importante
        reverse=True,
    )
    fi_path = os.path.join(MODEL_DIR, 'feature_importance.json')
    with open(fi_path, 'w') as f:
        json.dump(fi, f, indent=2)
    print("Feature importances (decrescente):")
    for item in fi:
        bar = '█' * int(item['importance'] * 40)  # Barra ASCII proporcional
        print(f"  {item['feature']:10s}  {item['importance']:.4f}  {bar}")

    # ---- 8. Serializar modelo e lista de colunas ----
    # colunas.pkl garante que a API use exatamente as mesmas features e ordem
    joblib.dump(modelo, caminho_modelo_saida)
    joblib.dump(col_modelo, os.path.join(MODEL_DIR, 'colunas.pkl'))
    print(f"Modelo salvo em {caminho_modelo_saida}")
    print("Treinamento concluído com sucesso!")


if __name__ == "__main__":
    # Ponto de entrada quando executado diretamente via `python train.py`
    treinar_modelo(os.path.join(MODEL_DIR, 'modelo_defasagem.pkl'))
