import cv2
import pickle
import numpy as np


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognize = cv2.face.EigenFaceRecognizer_create()
recognize.read("train/eigennednew.yml")

# labels = {"person_name": 1}
with open("pickle/labeleigennew.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Img = cv2.equalizeHist(gray)
    faces = faceCascade.detectMultiScale(Img, 1.3, 5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #ycord_start, ycord_end
        roi_color = img[y:y+h, x:x+w]
        gray_face = cv2.resize(roi_gray, (110, 110))

        id_, conf = recognize.predict(cv2.resize(roi_gray, (300, 300)))
        if conf >= 45: #and conf <=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(img, name, (x, y - 40), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "test.png"
            # cv2.imwrite(img_item, roi_gray)

        # gray_face = cv2.resize((gray[y:y+h, x:x+w]), (110, 110))
        # eyes = eyeCascade.detectMultiScale(gray_face)
        color = (255, 0, 0) #BGR 0-255
        stroke = 3
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)


    cv2.imshow("eigenface system", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


