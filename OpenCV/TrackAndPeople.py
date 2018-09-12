# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

#width = 480
cap = cv2.VideoCapture('emma.mov')
#ret, frame = cap.read()
#capcrop = imutils.resize(frame[0:1600,0:700], width = width)
#height, width = capcrop.shape[:2]
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('emmaout.mov',fourcc, 5.0, (height, width))

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
ret = True
while(ret):
    ret, frame = cap.read()
    image = imutils.resize(frame[0:900,0:1200], width = 480)   
    #image = frame 
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    #filename = imagePath[imagePath.rfind("/") + 1:]
    #print("[INFO] {}: {} original boxes, {} after suppression".format(
    #    filename, len(rects), len(pick)))
    # show the output images
    #cv2.imshow("Before NMS", orig)
    kernel = np.ones((11,3),np.float32)/33
    image = cv2.filter2D(image,-1,kernel)
    image = 255-image
    inputImageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(inputImageGray,50,200,apertureSize = 3)
    minLineLength = 50
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
    if lines is not None:
        print(lines)
        for x in range(0, 1):#len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
                pts = np.array([[180, 360], [x2 , y2]], np.int32)
                cv2.polylines(image, [pts], True, (0,255,0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,"Tracks Detected", (500, 250), font, 0.5, 255)
    cv2.imshow("Trolley_Problem_Result", image)
    cv2.imshow('edge', edges)
    #cv2.imshow("After NMS", image)
    #out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
#out.release()
cv2.destroyAllWindows()