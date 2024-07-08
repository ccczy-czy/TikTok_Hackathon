from flask import Flask, jsonify
from flask_cors import CORS

# app instance
app = Flask(__name__)
CORS(app)


# /api/home
@app.route("/api/home", methods=["GET"])
def home():
    return jsonify({"message": "Hello world!"})


if __name__ == "__main__":
    # remove debug=True if deploying to production
    app.run(debug=True, port=8080)
