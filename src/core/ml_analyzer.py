"""Módulo de inferência ML para classificação de URLs."""

import json
import os
import joblib
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "modelo_phishing.joblib")
_META_PATH = os.path.join(_BASE_DIR, "modelo_metadata.json")
DEFAULT_FEATURE_VALUE = -1


def _load_model() -> tuple:
    """Carrega pipeline e metadados do disco.
    Retorna: (pipeline, metadata, feature_order) ou (None, None, []).
    """
    try:
        pipeline = joblib.load(_MODEL_PATH)
        with open(_META_PATH, 'r') as f:
            metadata = json.load(f)
        return pipeline, metadata, metadata['features']
    except FileNotFoundError as e:
        print(f"ERRO: {e}. Execute o script de treinamento.")
        return None, None, []


modelo, metadata, FEATURE_ORDER = _load_model()


def _build_input(features_ml: dict, feature_order: list[str]) -> pd.DataFrame:
    """Monta DataFrame de input preenchendo ausentes com default.
    Requer: dict de features e ordem esperada pelo modelo.
    Retorna: DataFrame com 1 linha.
    """
    row = {feat: features_ml.get(feat, DEFAULT_FEATURE_VALUE) for feat in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def analyze_with_ml(features_ml: dict) -> dict:
    """Classifica URL como phishing/legítima via modelo treinado.
    Requer: dict com features extraídas da URL.
    Retorna: dict com 'previsao' e 'confianca'.
    """
    if not modelo:
        return {"previsao": "erro_modelo_nao_carregado", "confianca": "0%"}

    try:
        input_df = _build_input(features_ml, FEATURE_ORDER)
        probabilities = modelo.predict_proba(input_df)
        prediction = modelo.predict(input_df)

        classes = list(modelo.classes_)
        confidence = float(probabilities[0][classes.index(prediction[0])])

        return {"previsao": prediction[0], "confianca": f"{confidence:.0%}"}
    except Exception as e:
        print(f"Erro na predição: {e}")
        return {"previsao": "erro_na_predicao", "confianca": "0%"}
