from flask import Flask, request, jsonify, Response
from model import stroke_predictor

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exception(e: Exception) -> tuple[Response, int]:
    return jsonify({"error": str(e)}), 400

@app.route("/health")
def get_health() -> Response:
    return jsonify({
        "message": "API is working!"
    })

@app.route("/v1/predict", methods=["POST"])
def predict() -> Response:
    body = request.get_json()
    return jsonify(stroke_predictor.predict(body))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
