import logging
import os

from flask import request, Flask, jsonify, Response, send_from_directory

import dotenv

dotenv.load_dotenv()
app = Flask(__name__)

logger = logging.getLogger(__name__)

@app.route('/ping')
def ping():
    logger.info('ping received')
    return 'pong kek'


if __name__ == '__main__':
    app.run()