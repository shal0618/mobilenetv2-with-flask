from flask import Flask, render_template, request, Response, redirect, url_for, Markup, jsonify
from flask_bootstrap import Bootstrap
import shutil
import torch
import torch.nn as nn

import os
import cv2
import sys

sys.path.append("../")
from vision.nets.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.config import cfg


app = Flask(__name__)
net_type = 'mb2-ssd-lite'
model_path = cfg.model_path
label_path = cfg.label_path
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
output_dir = 'output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
Bootstrap(app)

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
webcamid = 0
cap = cv2.VideoCapture(webcamid)

img_array = []
@app.route('/capture')
def getFrames(get):
    """
    webカメラから受け取った動画をフレームごとに分割して推論、出力します。
    args:
        get: ボタンから受け取った値が格納されています。{true, false}
    """
    path = os.path.join(output_dir, 'output.mp4')
    size = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, 5.0, (size))
    get = get
    while True:
        ret, image = cap.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f'{class_names[labels[i]]}: {probs[i]:.2f}'
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, label,
                        (box[0] + 10, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        h, w, layers = image.shape
        size = (w, h)
        img_array.append(image)
        if get == 'true':
            print('start')
            video = cv2.VideoWriter(path, fourcc, 5.0, (size))
            img_array.clear()
            get = None
        elif get == 'false':
            print('stop')
            print(len(img_array))
            for i in range(len(img_array)):
                video.write(img_array[i])
            video.release()
            img_array.clear()
            print('video expoted', path)
            get = None
        if len(img_array) > cfg.img_array_limmit:
            img_array.clear()
            print('img_array cleared.')

        ret, jpg = cv2.imencode('test.jpg', image)
        yield b'--boundary\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tostring() + b'\r\n\r\n'


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/video_feed', methods=['GET'])
def video_feed():
    get = request.args.get('data')
    return Response(getFrames(get), mimetype='multipart/x-mixed-replace;boundary=boundary')


import webbrowser
webbrowser.get().open('http://localhost:5000')

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)