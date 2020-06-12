##########################################
import numpy as np
import cv2 as cv
import requests
from tensorflow.keras.models import load_model
##########################################

##########################################
width = 640
height = 480
threshold = 0.90
url = "http://192.168.42.129:8080/shot.jpg"
##########################################

def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img

# Loading the Model
model = load_model("models/final_model.h5")


# Testing the model by Android Camera
while(True):
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)

    img = cv.imdecode(img_arr, -1)
    # cv.imshow("Android Cam", img)

    img_ = np.asarray(img)

    img_ = cv.resize(img_, (28, 28))
    img_ = preProcessing(img_)
    img_.astype(np.float32)
    # cv.imshow("Processed Image", img)
    img_ = img_.reshape(1, 28, 28, 1)

    # Predicting the class
    classIndex = int(model.predict_classes(img_))
    # print(classIndex)

    prediction = model.predict(img_)
    # print(prediction)

    # probVal = np.amax(prediction)
    probVal = np.argmax(prediction, axis=-1)
    # print(probVal)

    if probVal > threshold:
        cv.putText(img, str(classIndex) + " " + str(probVal), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                   1)

    cv.imshow("Original Image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
