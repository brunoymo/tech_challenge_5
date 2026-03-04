import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix, roc_curve,
)
from sklearn.preprocessing import label_binarize
import joblib
from preprocessing import carregar_dados, preprocessar_pipeline, COLUNAS_MODELO

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '../data/raw')
MODEL_DIR  = os.path.join(BASE_DIR, '../model')


def treinar_modelo(caminho_modelo_saida: str):
    """
    Treina o modelo de ML, avalia e salva todos os artefatos em backend/model/.
    Artefatos gerados:
      - modelo_defasagem.pkl
      - colunas.pkl
      - metricas.json  (métricas completas)
      - feature_importance.json
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Carregando dados...")
    df_bruto = carregar_dados(
        os.path.join(DATA_DIR, 'dataset_2022.csv'),
        os.path.join(DATA_DIR, 'dataset_2023.csv'),
        os.path.join(DATA_DIR, 'dataset_2024.csv'),
    )

    print("Pré-processando dados...")
    df_proc = preprocessar_pipeline(df_bruto)

    # Manter apenas colunas do modelo + RISCO
    col_modelo = [c for c in COLUNAS_MODELO if c in df_proc.columns]
    df_proc = df_proc[col_modelo + ['RISCO']].dropna()

    X = df_proc[col_modelo]
    y = df_proc['RISCO']

    print(f"Dataset: {len(X)} amostras | Risco=1: {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Treinando RandomForestClassifier...")
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    # ---- Avaliação ----
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    roc_auc  = roc_auc_score(y_test, y_prob)
    acc      = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    rec      = recall_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")

    # ---- Salvar métricas ----
    metricas = {
        "accuracy":        round(acc, 4),
        "roc_auc":         round(roc_auc, 4),
        "f1":              round(f1, 4),
        "precision":       round(prec, 4),
        "recall":          round(rec, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": [round(v, 4) for v in fpr.tolist()],
            "tpr": [round(v, 4) for v in tpr.tolist()],
        },
        "n_amostras":      int(len(X)),
        "n_treino":        int(len(X_train)),
        "n_teste":         int(len(X_test)),
    }
    metricas_path = os.path.join(MODEL_DIR, 'metricas.json')
    with open(metricas_path, 'w') as f:
        json.dump(metricas, f, indent=2)
    print(f"Métricas salvas em {metricas_path}")

    # ---- Feature importances ----
    importances = modelo.feature_importances_
    fi = sorted(
        [{"feature": col, "importance": round(float(imp), 4)}
         for col, imp in zip(col_modelo, importances)],
        key=lambda x: x["importance"], reverse=True,
    )
    fi_path = os.path.join(MODEL_DIR, 'feature_importance.json')
    with open(fi_path, 'w') as f:
        json.dump(fi, f, indent=2)
    print("Feature importances:")
    for item in fi:
        print(f"  {item['feature']:10s}  {item['importance']:.4f}")

    # ---- Salvar modelo e colunas ----
    joblib.dump(modelo, caminho_modelo_saida)
    joblib.dump(col_modelo, os.path.join(MODEL_DIR, 'colunas.pkl'))
    print(f"Modelo salvo em {caminho_modelo_saida}")
    print("Treinamento concluído com sucesso!")


if __name__ == "__main__":
    treinar_modelo(os.path.join(MODEL_DIR, 'modelo_defasagem.pkl'))
