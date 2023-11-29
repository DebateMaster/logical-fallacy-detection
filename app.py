import logging
import os

from flask import request, Flask, jsonify, Response, send_from_directory

import server
import network

server.setup_logging.setup()
app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route('/text_predict')
# @server.check_api_token_header # TODO
def get_predict_handler() -> Response:
    logger.info('Recieved GET request')
    text = request.args.get('text')
    prediction = network.predict(text)
    return jsonify(prediction=prediction)


if __name__ == '__main__':
    app.run()