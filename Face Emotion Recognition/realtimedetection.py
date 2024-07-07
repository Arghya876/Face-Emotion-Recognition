import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")  # Fixed typo: "emotiondetectior" to "emotiondetector"
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            face_image = gray[q:q + s, p:p + r]  # Fixed variable name: images to face_image
            cv2.rectangle(im, (p, q), (p + r, q + s), (0, 255, 0), 2)
            face_image = cv2.resize(face_image, (48, 48))  # Fixed variable name: image to face_image
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]  # Fixed variable name: prediction_label
            cv2.putText(im, '%s' % prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255))
        cv2.imshow("Output", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()
