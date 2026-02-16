# URLScan API

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Security](https://img.shields.io/badge/Security-Phishing%20Detection-red.svg)](#)

API REST para deteccao de phishing em URLs utilizando Machine Learning. O servico extrai mais de 50 features de uma URL, incluindo padroes estruturais, caracteristicas de dominio e sinais do conteudo da pagina, e classifica como **phishing** ou **legitima** atraves de um modelo Gradient Boosting.

---

## Funcionalidades

- Classificacao de URLs como phishing ou legitima com score de confianca
- 53 features engenheiradas: estrutura da URL, dominio e conteudo HTML
- Endpoint para analise individual e em lote
- Treinamento com K-Fold cross-validation e tuning via GridSearchCV

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.12+ |
| Framework Web | Flask |
| ML | scikit-learn (GradientBoostingClassifier) |
| Extracao de Features | tldextract, BeautifulSoup4, Requests |
| Dados | pandas, NumPy, PyArrow |
| Gerenciador de Pacotes | uv |

---

## Como Usar

### Pre-requisitos

- Python 3.12 ou superior
- [uv](https://docs.astral.sh/uv/) (recomendado) ou pip

### Instalacao

1. Clone o repositorio:

```bash
git clone https://github.com/jpaullopes/URLScan-API.git
cd URLScan-API
```

1. Crie o ambiente virtual e instale as dependencias:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

1. Baixe o dataset do Kaggle:

   [Phishing URL Dataset](https://www.kaggle.com/datasets/hemanthpingali/phishing-url) -- baixe `Training.parquet` e `Testing.parquet` e coloque em `src/data/`.

2. Treine o modelo:

```bash
python src/core/train_model.py
```

1. Inicie o servidor:

```bash
flask --app app run --host 0.0.0.0 --port 5000
```

A API estara disponivel em `http://localhost:5000`.

---

## Exemplo de Uso

Para analisar uma URL, envie uma requisicao POST para o endpoint `/analyze`. Se quiser analisar em lote, envie uma requisicao POST para o endpoint `/analyze/batch`.

**Exemplo:**

```bash
curl -X POST http://localhost:5000/analyze \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.google.com"}'
```

**Resposta:**

```json
{
  "url": "https://www.google.com",
  "analise_ml": {
    "previsao": "legitimate",
    "confianca": "98%"
  }
}
```

---

## Licenca

Este projeto esta licenciado sob a licenca MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
