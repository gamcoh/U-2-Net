from flask import Flask, request
from base64 import b64decode
import logging
import u2net_lib as u2net

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/')
def root():
    return 'hi, what are you doing here?'

@app.route('/postImage/', methods=['POST'])
def applyU2OnImg():
    filename = request.form['filename']
    filepath = 'input_images/' + filename
    image = request.form['image']
    with open(filepath, 'wb') as f:
        f.write(b64decode(image))
    newImgPath = u2net.main(colored=True, imagepath=filepath)

    return open(newImgPath, 'rb').read()
