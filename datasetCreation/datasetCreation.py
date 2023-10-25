import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from utils import changeReferencePoint, normalizeCoords

# def changeReferencePoint(pointCoords, newRefPointCoords):
#     return [pointCoords[0] - newRefPointCoords[0], pointCoords[1] - newRefPointCoords[1]]
#
#
# def normalizeCoords(pointCoords, normFactorX, normFactorY):
#     return [pointCoords[0] / normFactorX, pointCoords[1] / normFactorY]


cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# 1 (code == 49) -> for gesture 1 (first gesture)
# 2 (code == 50) -> for gesture 2 (second gesture)
# 0 (code == 48) -> for other gestures
# r (code == 114) -> for stopping data recording

recordGroundTruthFlag = False
groundTruth = -1
coordsToWrite = []

while cap.isOpened():

    # receive image from camera
    ret, img = cap.read()

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    img.flags.writeable = False
    # flip image horizontally
    img = cv2.flip(img, flipCode=1)

    img_height, img_width, img_depth = img.shape

    imgRGB = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    # process teh image with the hands detector
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

        # cv2.rectangle(img=img, pt1=(xOfLeftMost, yOfUpMost), pt2=(xOfRightmost, yOfDownmost), color=(0,255,0))
        cv2.rectangle(img=img, pt1=(rectOriginX, rectOriginY), pt2=(rectOriginX + rectWidth, rectOriginY + rectHeight), color=(0, 255, 0), thickness=2)
        # print("Point 0 original coords = (" + str(xyPixelCoords[0][0]) + " , " + str(xyPixelCoords[0][1]) + ")")
        # print("Rectangle origin coords = (" + str(rectOriginX) + " , " + str(rectOriginY) + ")")
        # print("Point 0 new coords = (" + str(newXyPixelCoords[0][0]) + " , " + str(newXyPixelCoords[0][1]) + ")")
        # print("Point 0 normalized coords = (" + str(normalizedXyPixelCoords[0][0]) + " , " + str(normalizedXyPixelCoords[0][1]) + ")")

        if recordGroundTruthFlag:
            flattenedCoords = list(np.array(normalizedXyPixelCoords).flatten())
            flattenedCoords.append(groundTruth)
            coordsToWrite.append(flattenedCoords)

    None
    # show image in window
    cv2.imshow("window", img)

    keyCode = cv2.waitKey(10)

    if keyCode == 49:
        print("Pressed " + chr(keyCode) + ", start writing ground truth for first gesture.")
        recordGroundTruthFlag = True
        groundTruth = 1
    if keyCode == 50:
        print("Pressed " + chr(keyCode) + ", start writing ground truth for second gesture.")
        recordGroundTruthFlag = True
        groundTruth = 2
    if keyCode == 48:
        print("Pressed " + chr(keyCode) + ", start writing ground truth for other gestures.")
        recordGroundTruthFlag = True
        groundTruth = 0
    if keyCode == 114:
        print("Pressed " + chr(keyCode) + ", save ground truth to file and stop writing ground truth.")
        recordGroundTruthFlag = False
        data = pd.DataFrame(coordsToWrite)
        # save to .csv
        data.to_csv(path_or_buf='C:\\Users\\Nikos\\PycharmProjects\\audioOutputSwitch\\data\\' + str(groundTruth) + '.csv', header=None, index=False)
        groundTruth = -1
        coordsToWrite.clear()
        data = pd.DataFrame()