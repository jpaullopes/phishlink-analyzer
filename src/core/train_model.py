"""Pipeline de treinamento com K-Fold e tuning para detecção de phishing."""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, "data")
OUTPUT_MODEL = os.path.join(_BASE_DIR, "modelo_phishing.joblib")
OUTPUT_METADATA = os.path.join(_BASE_DIR, "modelo_metadata.json")
N_FOLDS = 5
RANDOM_STATE = 42

FEATURES_URL = [
    'length_url', 'length_hostname', 'nb_dots', 'nb_hyphens',
    'nb_at', 'nb_qm', 'nb_and', 'nb_eq', 'nb_underscore',
    'nb_slash', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path',
    'https_token', 'ratio_digits_url', 'ratio_digits_host',
    'punycode', 'port', 'tld_in_path', 'tld_in_subdomain',
    'abnormal_subdomain',
]

FEATURES_DOMAIN = [
    'nb_subdomains', 'prefix_suffix', 'shortening_service',
    'nb_redirection', 'nb_external_redirection', 'char_repeat',
    'shortest_word_host', 'longest_word_host', 'avg_word_host',
    'avg_word_path', 'phish_hints', 'suspecious_tld',
    'random_domain', 'ip',
]

FEATURES_CONTENT = [
    'nb_hyperlinks', 'ratio_intHyperlinks',
    'ratio_extHyperlinks', 'ratio_nullHyperlinks',
    'nb_extCSS', 'login_form', 'external_favicon',
    'links_in_tags', 'submit_email', 'sfh', 'iframe',
    'popup_window', 'safe_anchor', 'onmouseover', 'right_clic',
    'empty_title', 'domain_in_title',
]

ALL_FEATURES = FEATURES_URL + FEATURES_DOMAIN + FEATURES_CONTENT


def load_and_combine_data(data_dir: str) -> pd.DataFrame:
    """Carrega Training + Testing e combina em um dataset único.
    Requer: caminho do diretório com os parquets.
    Retorna: DataFrame combinado.
    """
    df_train = pd.read_parquet(f"{data_dir}/Training.parquet")
    df_test = pd.read_parquet(f"{data_dir}/Testing.parquet")
    return pd.concat([df_train, df_test], ignore_index=True)


def filter_available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Filtra features que realmente existem no dataset.
    Requer: DataFrame e lista de features desejadas.
    Retorna: lista com apenas as features encontradas.
    """
    return [f for f in features if f in df.columns]


def build_pipeline() -> Pipeline:
    """Cria pipeline Imputer → Scaler → GradientBoosting.
    Retorna: Pipeline sklearn pronto para fit.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def run_cross_validation(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict:
    """Executa K-Fold cross-validation com múltiplas métricas.
    Requer: pipeline, features (X), target (y), estratégia de CV.
    Retorna: dict com média, desvio e valores por fold de cada métrica.
    """
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1,
    )

    metrics = {}
    for metric in scoring:
        key = metric.replace('_weighted', '')
        values = results[f'test_{metric}']
        metrics[key] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'per_fold': [float(v) for v in values],
        }
    return metrics


def run_hyperparameter_tuning(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> GridSearchCV:
    """Busca os melhores hiperparâmetros via GridSearchCV.
    Requer: pipeline, features (X), target (y), estratégia de CV.
    Retorna: GridSearchCV já fitado (best_estimator_ disponível).
    """
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
    }

    grid = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring='f1_weighted', n_jobs=-1, verbose=1, refit=True,
    )
    grid.fit(X, y)
    return grid


def get_feature_importance(pipeline: Pipeline, features: list[str], top_n: int = 20) -> list[dict]:
    """Extrai ranking de importância das features do modelo.
    Requer: pipeline treinado e lista de nomes das features.
    Retorna: lista de dicts [{feature, importance}] ordenada.
    """
    classifier = pipeline.named_steps['classifier']
    if not hasattr(classifier, 'feature_importances_'):
        return []

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    return [
        {'feature': features[idx], 'importance': round(float(importances[idx]), 4)}
        for idx in indices[:top_n]
    ]


def save_model_and_metadata(pipeline: Pipeline, features: list[str], cv_metrics: dict, best_params: dict, ranking: list[dict]) -> None:
    """Salva pipeline treinado e metadados em disco.
    Requer: pipeline, features, métricas, params e ranking.
    """
    joblib.dump(pipeline, OUTPUT_MODEL)

    metadata = {
        'features': features,
        'best_params': best_params,
        'cv_metrics': {k: {'mean': round(v['mean'], 4), 'std': round(v['std'], 4)} for k, v in cv_metrics.items()},
        'feature_importance': ranking,
        'n_features': len(features),
        'model_type': 'GradientBoostingClassifier',
        'n_folds': N_FOLDS,
    }

    with open(OUTPUT_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    df = load_and_combine_data(DATA_DIR)
    features = filter_available_features(df, ALL_FEATURES)
    X = df[features]
    y = df['status']

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"Dataset: {X.shape[0]} amostras, {len(features)} features")
    print(f"Distribuição: {dict(y.value_counts())}")

    # Cross-validation
    print(f"\n--- Cross-Validation ({N_FOLDS} folds) ---")
    pipeline = build_pipeline()
    cv_metrics = run_cross_validation(pipeline, X, y, cv)

    for name, vals in cv_metrics.items():
        print(f"  {name:<12}: {vals['mean']:.4f} (±{vals['std']:.4f})")

    # Tuning
    print(f"\n--- Hyperparameter Tuning (GridSearchCV) ---")
    grid = run_hyperparameter_tuning(build_pipeline(), X, y, cv)

    best_params = {k.replace('classifier__', ''): v for k, v in grid.best_params_.items()}
    print(f"  Melhores params: {best_params}")
    print(f"  F1 cross-validated: {grid.best_score_:.4f}")

    best_pipeline = grid.best_estimator_

    # Feature importance
    print(f"\n--- Feature Importance (Top 20) ---")
    ranking = get_feature_importance(best_pipeline, features)
    for i, item in enumerate(ranking):
        print(f"  {i+1:2d}. {item['feature']:<30} {item['importance']:.4f}")

    # Salvar
    save_model_and_metadata(best_pipeline, features, cv_metrics, best_params, ranking)
    print(f"\nModelo salvo: {OUTPUT_MODEL}")
    print(f"Metadados salvos: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
