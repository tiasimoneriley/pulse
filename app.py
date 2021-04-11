import json
import os

from flask import Flask, render_template
from flask_sockets import Sockets

import model

app = Flask(__name__)
sockets = Sockets(app)


@app.route("/")
def index():
    return render_template("splash.html")


@app.route("/begin")
def get_heartrate():
    return render_template("index.html")


@sockets.route('/echo')
def echo_socket(ws):
    while True:
        message = json.loads(ws.receive())
        signals = model.parse_RGB(message)

        ws.send(signals)


if __name__ == "__main__":
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    port = int(os.environ.get('PORT', 5000))
    print("Hosting on port {}".format(port))
    server = pywsgi.WSGIServer(('', port), app, handler_class=WebSocketHandler)
    server.serve_forever()
