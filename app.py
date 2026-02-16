"""API Flask para análise de URLs via ML."""

from flask import Flask, request, jsonify
from src.core.phishing_analyzer import PhishingAnalyzer
from src.core.ml_analyzer import analyze_with_ml

app = Flask(__name__)
phishing_analyzer = PhishingAnalyzer()


@app.route("/analyze", methods=["POST"])
def analyze_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL não fornecida"}), 400

    url = data["url"]
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    ml_features = phishing_analyzer.analyze(url)["ml_features"]
    ml_result = analyze_with_ml(ml_features)

    return jsonify({
        "url": url,
        "analise_ml": ml_result,
    })

@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    data = request.get_json()
    urls = data.get("urls", [])
    if not urls:
        return jsonify({"error": "Lista de 'urls' é obrigatória e não pode estar vazia"}), 400

    results = []
    for url in urls:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        ml_features = phishing_analyzer.analyze(url)["ml_features"]
        ml_result = analyze_with_ml(ml_features)
        results.append({
            "url": url,
            "analise_ml": ml_result,
        })

    return jsonify({
        "results": results,
    })  


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
