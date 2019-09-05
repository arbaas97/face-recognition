import cv2
import os
import numpy as np
from PIL import Image
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "webcamnew")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognize = cv2.face.EigenFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files, in os.walk(image_dir):
    for file in files :
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ","-" ).lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print(label_ids)

            pil_image = Image.open(path).convert("L")
            # image_array = np.array(pil_image, "uint8")
            size = (500, 500)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            # cv2.imshow("photo", image_array)
            #print(image_array)
            # scalefactor pengaruh radius
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.4, minNeighbors=1, minSize= (30,30))
            for(x,y,w,h) in faces:
                # a = img.resize(280,280)
                roi = img[y:y+h, x:x+w]
                x_train.append(cv2.resize(roi,(280,280)))
                y_labels.append(id_)

with open("pickle/labeleigennew.pickle", "wb") as f:
    pickle.dump(label_ids, f)
recognize.train(x_train, np.array(y_labels))
recognize.save("train/eigennednew.yml")
print("train success")