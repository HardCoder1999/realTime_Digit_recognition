import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

##################################################
width = 640
height = 480
threshold = 0.90
##################################################

def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img
# Loading the Model
model = load_model('models/final_model.h5')

# Setting up the WebCamera for Input
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while(True):
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)

    img = cv.resize(img, (28, 28))
    img = preProcessing(img)
    img.astype(np.float32)
    #cv.imshow("Processed Image", img)
    img = img.reshape(1, 28, 28, 1)

    # Predicting the class
    classIndex = int(model.predict_classes(img))
    # print(classIndex)

    prediction = model.predict(img)
    # print(prediction)

    # probVal = np.amax(prediction)
    probVal = np.argmax(prediction, axis=-1)
    # print(probVal)

    if probVal > threshold:
        cv.putText(imgOriginal, str(classIndex) + " " + str(probVal), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                   1)

    cv.imshow("Original Image", imgOriginal)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
