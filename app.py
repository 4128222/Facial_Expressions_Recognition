from flask import Flask, make_response, jsonify, abort, request
from scipy import misc
from discern import discern_face
from keras.models import load_model
from keras.backend import clear_session
from classify import classify_one_image

import os
import pandas as pd
import tensorflow as tf
import face_recognition

app = Flask(__name__)

KIND_LISTS = ['Fear','Anger', 'Normal', 'Happy', 'Sad', 'Surprised']

# ValueError: Tensor Tensor("dense_2/Softmax:0", shape=(?, 6), dtype=float32) is not an element of this graph.
# https://cloud.tencent.com/developer/article/1167171
'''
graph = None
model = None
def load():
    global graph
    graph = tf.get_default_graph()
 
    global model
    model = load_model('models/face_top_model.h5')
'''
@app.route('/face/1.0/upload', methods=['post'])
def upload():
    if 'file' not in request.files:
        abort(400)
    files = request.files['file']
    # 读取图片
    image = face_recognition.load_image_file(files)
    # 识别面部，并对面部进行裁剪
    image = discern_face(image)
    if image is None:
        abort(401)
    # 对表情进行分类
    # with graph.as_default():
    result = classify_one_image(image, (64,64))
    label = KIND_LISTS[result[0]]
    # a = result[2][0].tolist()
    expression = {
        'code': 200,
        'msg': "Success",
        'result': label,
        'probability': result[2][0].tolist()
    }
    return jsonify({'expression': expression})
@app.errorhandler(400)
def invalid_request(error):
    expression = {
        'code': 400,
        'msg': "Invalid request",
        'result': None,
        'probability': None
    }
    return make_response(jsonify({'error': expression}), 400)

@app.errorhandler(401)
def noface_request(error):
    expression = {
        'code': 401,
        'msg': "No face",
        'result': None,
        'probability': None
    }
    return make_response(jsonify({'error': expression}), 401)

if __name__ == '__main__':
    # load()
    app.debug = False
    # app.run(host='0.0.0.0:8001')
    app.run(debug=False)
