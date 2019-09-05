import cv2
import random
import numpy as np
import os
from PIL import Image
import pickle

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')# haarcascade default.
cap = cv2.VideoCapture(0)
count = 0

id = input("masukkan id: ")
name = input("masukkan nama:")

try:
    # Create target Directory
    os.mkdir("webcam/" + str(name))  # membuat direktori
    print("Directory ", "webcam/" + str(name), " Created ")

except FileExistsError:
    print("Directory ", "webcam/" + str(name), " already exists")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edge = cv2.cvtColor(frame, cv2.COLOR_+++BGR2HSV_FULL, 60, 60)
    faces = face.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=2, minSize=(10,10))
    for (x, y, w, h) in faces:
        # if cv2.waitKey(2) & 0xFF == ord('s'):
        count += 1
        # Capture foto sebanysk 15 sample
        cv2.imwrite("webcam/"+name+"/user."+id+"."+str(count)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.waitKey(100)

    cv2.imshow('Frame', frame)

    cv2.waitKey(1)
    if count > 15:
        break

cap.release()
cv2.destroyAllWindows()