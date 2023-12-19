from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    try:
        text = request.get_json().get("message")
        if text is None:
            return jsonify({"error": "Invalid or missing 'message' in request JSON"}), 400

        # TODO: check if text is valid
        response = get_response(text, model, intents, all_words, tags, device)
        message = {"answer": response}
        return jsonify(message)
    except Exception as e:
        # Log the error for debugging
        print(f"Error in predict route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
