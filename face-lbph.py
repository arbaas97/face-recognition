import numpy as np
import cv2
import imutils.paths as paths
import pickle
import os
from PIL import Image

face_detect = 'haarcascade_frontalface_alt.xml'

def start():
	print('What do you want to do?')
	print('1. CreateData 2.train 3.RecogData')

	set = input()
	if set == "1":
		dataSet_create()
	elif set == "2":
		train()
	elif set== "3":
		recognition()

def dataSet_create():
	face_cascade = cv2.CascadeClassifier(face_detect)
	cap = cv2.VideoCapture(0)

	# path = "F:\\opencv\\face recognisation\\dataset\\"  # path were u want store the data set
	path = "images\\"
	print("Enter Your Name :")
	name = input()

	try:
		# Create target Directory
		os.mkdir(path + str(name))
		print("Directory ", path + str(name), " Created ")

	except FileExistsError:
		print("Directory ", path + str(name), " already exists")

	sampleN = 0

	while 1:
		ret, frame = cap.read()
		# frame = img.copy()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.4, 2)
		for x, y, w, h in faces:
			cv2.imwrite(path + str(name) + "/" + str(sampleN) + ".jpg", gray[y:y + h, x:x + w])
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
			sampleN = sampleN + 1
			cv2.waitKey(200)

		cv2.imshow('img', frame)

		cv2.waitKey(1)
		if sampleN > 29:
			break

	cap.release()
	cv2.destroyAllWindows()
	start()

def train():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	image_dir = os.path.join(BASE_DIR, "images")

	#Training untuk metode LBPH
	face_cascade = cv2.CascadeClassifier(face_detect)
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	current_id = 0
	label_ids = {}
	y_labels = []
	x_train = []

	#Membaca Image yang ada pada direktori (path)
	for root, dirs, files, in os.walk(image_dir):
		for file in files :
			if file.endswith("png") or file.endswith("jpg"):
				path = os.path.join(root, file)
				label = os.path.basename(root).replace(" ","-" ).lower()
				#print(label, path)
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1 # memberikan nama setiap folder
				id_ = label_ids[label]
				#print(label_ids)
				pil_image = Image.open(path).convert("L")
				# image_array = np.array(pil_image, "uint8")
				size = (500, 500)
				final_image = pil_image.resize(size, Image.ANTIALIAS)# Resize image menjadi (500, 500)
				image_array = np.array(final_image, "uint8")
				# cv2.imshow("photo", image_array)
            	#print(image_array)
				faces = face_cascade.detectMultiScale(image_array, 1.4, 1)#scalefactor pengaruh radius
				for(x,y,w,h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_train.append(roi)
					y_labels.append(id_)


	with open("pickle/label.pickle", "wb") as f:
		pickle.dump(label_ids, f)
	recognizer.train(x_train, np.array(y_labels))
	recognizer.save("train/trainner.yml")
	start()

def recognition():
	face_cascade = cv2.CascadeClassifier(face_detect)
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("train/trainner.yml")

	# labels = {"person_name": 1}
	with open("pickle/label.pickle", "rb") as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}

	cap = cv2.VideoCapture(0)

	while(True):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors= 2, minSize=(30,30))
		for (x, y, w, h) in faces:
			print(x, y, w, h)
			roi_gray = gray[y:y+h, x:x+w] #ycord_start, ycord_end
			roi_color = frame [y:y+h, x:x+w]

			id_, conf = recognizer.predict(roi_gray)
			if conf >= 55: #and conf <=85:
				print(conf)
				print(id_)
				print(labels[id_]) # {1:'arba'}
				font = cv2.FONT_HERSHEY_COMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				stroke = 1
				cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)


			color = (255, 0, 0) #BGR 0-255
			stroke = 3
			end_cord_x = x+w
			end_cord_y = y+h
			cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

		#display the resulting frame
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('s'):
			break


	cap.release()
	cv2.destroyAllWindows()
	start()

start()
