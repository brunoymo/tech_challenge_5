"""
evaluate.py
-----------
Avaliação detalhada do modelo treinado: métricas completas, matriz de confusão,
curva ROC e importância das features. Pode ser executado standalone ou importado
pela API ou pelo notebook de EDA.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '../model/modelo_defasagem.pkl')
COLS_PATH  = os.path.join(BASE_DIR, '../model/colunas.pkl')
METRICAS_PATH = os.path.join(BASE_DIR, '../model/metricas.json')
FEAT_IMP_PATH = os.path.join(BASE_DIR, '../model/feature_importance.json')


def carregar_modelo():
    """Carrega modelo e colunas do disco."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Execute train.py primeiro.")
    modelo  = joblib.load(MODEL_PATH)
    colunas = joblib.load(COLS_PATH)
    return modelo, colunas


def avaliar(X_test: pd.DataFrame, y_test: pd.Series, modelo) -> dict:
    """
    Calcula métricas completas a partir de conjuntos de teste.

    Retorna dicionário com:
    - accuracy, roc_auc, f1, precision, recall
    - confusion_matrix  (lista de listas)
    - classification_report (texto)
    - roc_curve: {fpr, tpr, thresholds}
    """
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    metricas = {
        "accuracy":        round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc":         round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1":              round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision":       round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":          round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
        "roc_curve": {
            "fpr": [round(v, 4) for v in fpr.tolist()],
            "tpr": [round(v, 4) for v in tpr.tolist()],
        },
    }
    return metricas


def feature_importance(modelo, colunas: list) -> list[dict]:
    """
    Retorna importância das features como lista ordenada de dicts
    {feature, importance}.
    """
    importances = modelo.feature_importances_
    fi = sorted(
        [{"feature": col, "importance": round(float(imp), 4)} for col, imp in zip(colunas, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return fi


def rodar_avaliacao_completa():
    """
    Executado standalone: recarrega dados, re-divide treino/teste e gera
    todos os artefatos de avaliação em backend/model/.
    """
    from preprocessing import carregar_dados, preprocessar_pipeline, COLUNAS_MODELO

    DATA_DIR = os.path.join(BASE_DIR, '../data/raw')
    print("Carregando dados...")
    df = carregar_dados(
        os.path.join(DATA_DIR, 'dataset_2022.csv'),
        os.path.join(DATA_DIR, 'dataset_2023.csv'),
        os.path.join(DATA_DIR, 'dataset_2024.csv'),
    )
    df = preprocessar_pipeline(df)

    # Filtrar apenas colunas do modelo + target
    col_modelo = [c for c in COLUNAS_MODELO if c in df.columns]
    df = df[col_modelo + ['RISCO']].dropna()

    X = df[col_modelo]
    y = df['RISCO']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo, colunas = carregar_modelo()
    X_test = X_test[colunas]

    print("Calculando métricas...")
    metricas = avaliar(X_test, y_test, modelo)

    print("\n=== Relatório de Classificação ===")
    print(metricas["classification_report"])
    print(f"ROC-AUC : {metricas['roc_auc']}")
    print(f"Accuracy: {metricas['accuracy']}")
    print(f"F1-Score: {metricas['f1']}")

    # Salvar métricas detalhadas
    metricas_para_salvar = {k: v for k, v in metricas.items() if k != 'classification_report'}
    with open(METRICAS_PATH, 'w') as f:
        json.dump(metricas_para_salvar, f, indent=2)
    print(f"Métricas salvas em {METRICAS_PATH}")

    # Feature importances
    fi = feature_importance(modelo, colunas)
    with open(FEAT_IMP_PATH, 'w') as f:
        json.dump(fi, f, indent=2)
    print("Feature importances:")
    for item in fi:
        bar = "█" * int(item["importance"] * 40)
        print(f"  {item['feature']:10s} {item['importance']:.4f}  {bar}")
    print(f"Importâncias salvas em {FEAT_IMP_PATH}")


if __name__ == "__main__":
    rodar_avaliacao_completa()
