from flask import Flask
from routes.recommend import recommend_bp
from utils.logging_config import setup_logger

setup_logger()
app = Flask(__name__)
app.register_blueprint(recommend_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
