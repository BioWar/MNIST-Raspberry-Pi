import tflite_runtime.interpreter as tflite
from flask import Response
from flask import Flask
from flask import render_template
from flask import session, redirect, url_for, flash, request
from flask_bootstrap import Bootstrap
from flask_moment import Moment
import numpy as np
import datetime
import time
import argparse
import threading
import imutils
import cv2

outputFrame = None
biggerOutputFrame = None
pooledFrame = None
lock = threading.Lock()

# Initialize a flask object
app = Flask(__name__)
app.config['SECRET_KEY'] = 'very secret key'
bootstrap = Bootstrap(app)
moment = Moment(app)

# Initialize keras_lite interpreter
interpreter = tflite.Interpreter(model_path="/home/admin/OpenCV_Keras/mnist_cnn_tflite_model.tflite")
interpreter.allocate_tensors()

# Input details (shape) for future data
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

# Video capturing from DroidCam over local WiFi network
# vs =  cv2.VideoCapture(f"http://192.168.0.103:4747/video")
vs = cv2.VideoCapture("http://192.168.31.71:4747/video")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route("/predict_number/", methods=['POST'])
def predict_number():
    number = predict_number()
    flash(f'Your number is: {number}')
    return render_template('index.html', message="[INFO] predicted number is {number}")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

def rotate(image, angle, center=None, scale=1.0):
    # Optional function to rotate an image
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def inverse_pooling(image, width, height):
    # Optional function to upscale an image
    poolingFrame = np.zeros((width, height))
    scale = width // image.shape[0]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            poolingFrame[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = image[i, j]
    return poolingFrame

def make_stream():
    global vs, outputFrame, biggerOutputFrame, lock
    while(True):

        # Capture frame-by-frame
        ret, frame = vs.read()
        # Optional rotation and flipping for smartphone camera image
        frame = rotate(frame, 90)
        frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        grayOrig = gray.copy()
        # Resize to be compatible with input shape for the model
        gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        ret, threshOrig = cv2.threshold(grayOrig, 120, 255, cv2.THRESH_BINARY)
        # Tried "inverse pooling" on output to create bigger picture from 28x28 pixels
        # poolingFrame = inverse_pooling(thresh, 28*15, 28*15)

        # Our operations on the frame come here
    	# Display the resulting frame
        with lock:
            outputFrame = thresh.copy()
            biggerOutputFrame = threshOrig.copy()
            # pooledFrame = poolingFrame.copy()

def generate():
    global outputFrame, biggerOutputFrame, pooledFrame, lock
    while True:
        with lock:
            (flag, encodedImage) = cv2.imencode(".jpg", biggerOutputFrame) # Change to pooledFrame or outputFrame for different picture
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')

def predict_number():
    global outputFrame, interpreter, input_details, output_details
    # (28, 28) -> (1, 28, 28, 1)
    input_data = np.array(outputFrame, dtype=np.float32)
    input_data = input_data[..., np.newaxis]
    input_data = input_data.reshape((1, 28, 28, 1))
    # Setting tensor in model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Predicting number
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f'[INFO] array: {output_data}')
    return np.argmax(output_data)

if __name__=="__main__":
    # When everything done, release the capture
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of server")
    args = vars(ap.parse_args())
    t = threading.Thread(target=make_stream)
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

vs.stop()
