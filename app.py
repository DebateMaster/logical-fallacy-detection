import logging
import os

from flask import request, Flask, jsonify, Response, send_from_directory, make_response

import server
import network

server.setup_logging.setup()
app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route('/text_predict', methods=['GET'])
def get_predict_handler():
    if request.method == 'OPTIONS':
        response = Flask.make_response()
    else:
        logger.info('Received GET request')
        text = request.args.get('text')
        prediction = network.predict(text)
        response = make_response(jsonify(prediction=prediction))
    
    # Preflight request handling
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, User-Agent'
    response.headers['Access-Control-Max-Age'] = '86400'
    response.headers['Access-Control-Request-Method'] = 'GET'
    
    return response
        

if __name__ == '__main__':
    app.run()