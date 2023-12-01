import logging
import os

from flask import request, Flask, jsonify, Response, send_from_directory, make_response

import server
import network

server.setup_logging.setup()
app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route('/text_predict', methods=['GET', 'OPTIONS'])
def get_predict_handler():
    if request.method == 'OPTIONS':
        # Preflight request handling
        response = Flask.make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    else:
        logger.info('Received GET request')
        text = request.args.get('text')
        prediction = network.predict(text)
        return jsonify(prediction=prediction)


if __name__ == '__main__':
    app.run()