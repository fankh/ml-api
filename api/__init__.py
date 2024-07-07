# api/__init__.py
from flask import Flask

def create_api():
    app = Flask(__name__)
    from .routes.init_routes import init as init_blueprint

    app.register_blueprint(init_blueprint, url_prefix = '/init')

    return app