from flask import Blueprint, jsonify, request

init = Blueprint('init', __name__)

@init.route('/hello', methods=['GET'])
def hello():
    return jsonify(message = 'Hello test app')