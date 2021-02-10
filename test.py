from keras.models import load_model
import numpy as np
import imutils
import cv2
from keras.models import load_model
from collections import deque
import pickle

# construct the argument parse and parse the arguments
model_path = r"E:\Wheat Disease Detection\activity.model"
input = r"E:\Wheat Disease Detection\Dataset\Leaf Rust\00161.jfif"
label = r"E:\Wheat Disease Detection\lb.pickle"

moodel = load_model(model_path)
lb = pickle.loads(open("label", "rb").read())

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)

vs = cv2.VideoCapture(input)

(W, H) = (None, None)

while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean

	preds = moodel.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)

	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	text = "PREDICTION: {}".format(label.upper())
	cv2.putText(output, text, (4, 4), cv2.FONT_HERSHEY_SIMPLEX,
		0.25, (200,255,155), 2)

	cv2.imshow("Output",output)
	key = cv2.waitKey(10) & 0xFF

	if key == ord("q"):
		break

vs.release()
