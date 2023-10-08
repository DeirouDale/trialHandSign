import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20
imgSize = 500

folder = "data/C"
counter = 0

labels = ["Phase 1", "Phase 2", "Phase 3"]
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w

            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape

                wGap = math.ceil((500 - wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(img)
                print(prediction, index)
            else:
                k = imgSize/w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize,  hCal))
                imgResizeShape = imgResize.shape

                hGap = math.ceil((500 - hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

    #turn off window
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

