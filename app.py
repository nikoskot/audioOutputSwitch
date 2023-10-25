import queue
import cv2
import mediapipe as mp
import numpy as np
from utils import *
import tensorflow as tf
from audioOutputSwitchViaPowershell.audioOutputSwitchViaPowershell import switchDefaultPlaybackDevice

# create opencv vide capture object
cap = cv2.VideoCapture(0)

# mediapipe hand recognition setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# possible classes
classes = ['Other', 'First', 'Second']

# load inference model
model = tf.keras.models.load_model('C:\\Users\\Nikos\\PycharmProjects\\audioOutputSwitch\\model\\modelCheckpoints\\classifierCheckpoint.hdf5')

# switch buffer
buffer = queue.Queue(maxsize=100)

while cap.isOpened():

    # receive image from camera
    ret, img = cap.read()

    if not(ret):
        print("No frame")
        continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    img.flags.writeable = False

    # flip image horizontally
    img = cv2.flip(img, flipCode=1)

    img_height, img_width, img_depth = img.shape

    imgRGB = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    # process the image with the hands detector
    results = hands.process(imgRGB)

    # make the image writeable
    img.flags.writeable = True

    # draw the landmarks
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image=img, landmark_list=handLandmarks, connections=mp_hands.HAND_CONNECTIONS)

    xyPixelCoords = []
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks[0].landmark:
            x = np.minimum(np.maximum(0, int(landmark.x * img_width)), img_width)
            y = np.minimum(np.maximum(0, int(landmark.y * img_height)), img_height)
            xyPixelCoords.append([x, y])

        xyPixelCoords = np.array(xyPixelCoords)

        # coordinates of bounding rectangle
        rectOriginX, rectOriginY, rectWidth, rectHeight = cv2.boundingRect(xyPixelCoords)

        # change reference point to origin of bounding box
        newXyPixelCoords = []
        for coords in xyPixelCoords:
            newXyPixelCoords.append(changeReferencePoint(coords, [rectOriginX, rectOriginY]))

        # normalize new coordinates by rectangle size
        normalizedXyPixelCoords = []
        for coords in newXyPixelCoords:
            normalizedXyPixelCoords.append(normalizeCoords(coords, rectWidth, rectHeight))

        # draw rectangle
        cv2.rectangle(img=img, pt1=(rectOriginX, rectOriginY), pt2=(rectOriginX + rectWidth, rectOriginY + rectHeight), color=(0, 255, 0), thickness=2)

        # convert coordinates to form suitable for inference
        coords = np.reshape(np.array(normalizedXyPixelCoords).flatten(), (-1, 42))

        # class inference
        prediction = model.predict(coords)
        prediction = np.argmax(prediction, axis=1)[0]

        # add prediction to buffer
        buffer = insertAtEndOfQueue(buffer, prediction)

        # check buffer content for switch condition
        if chechForSwitch(buffer):
            switchDefaultPlaybackDevice()
            print("Switch")
            buffer = clearQueue()

        cv2.putText(img=img, text=classes[prediction], org=(10, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=2)

    else:
        buffer = clearQueue()

    # show image in window
    # cv2.imshow("window", img)

    keyCode = cv2.waitKey(10)