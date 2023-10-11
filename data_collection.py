import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 500 #set image size

folder = "data/B"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #create white image
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #crop image

            imgCropShape = imgCrop.shape #get shape of crop image

            aspectRatio = h/w #get aspect ratio of crop image

            if aspectRatio > 1: #if height is greater than width
                k = imgSize/h #get ratio of image size to height
                wCal = math.ceil(k * w) #calculate width
                imgResize = cv2.resize(imgCrop, (wCal, imgSize)) #resize image
                imgResizeShape = imgResize.shape #get shape of resized image

                wGap = math.ceil((500 - wCal)/2) #calculate gap
                imgWhite[:, wGap:wCal+wGap] = imgResize #add image to white image
            else: #if width is greater than height
                k = imgSize/w #get ratio of image size to width
                hCal = math.ceil(k * h) #calculate height
                imgResize = cv2.resize(imgCrop, (imgSize,  hCal)) #resize image
                imgResizeShape = imgResize.shape #get shape of resized image

                hGap = math.ceil((500 - hCal)/2) #calculate gap
                imgWhite[hGap:hCal+hGap, :] = imgResize #add image to white image

            
            # cv2.imshow('ImageCrop', imgCrop) #show crop image
            cv2.imshow('ImageWhite', imgWhite) #show white image

    # cv2.imshow('Image', img) #show original image
    key = cv2.waitKey(1) #wait for key press
    if key == ord('s'): #if key is s
        counter += 1 #increment counter
        cv2.imwrite(f"{folder}/Image.{time.time()}.jpg", imgWhite) #save image
        print(counter) #print counter

    #turn off window
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

