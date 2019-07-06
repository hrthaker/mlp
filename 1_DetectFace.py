import cv2
import matplotlib.pyplot as plt


# Load classifier
face_classifier = cv2.CascadeClassifier("./files/haarcascade_frontalface_alt.xml")

# Capture video
cap = cv2.VideoCapture(0)
ds_factor=1.5
while True:
	# Read frames
	ret,frame = cap.read()
	frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_AREA)
	#Convert frame into gray scale image
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#Detect faces in the image
	faces = face_classifier.detectMultiScale(gray_frame,1.3,5)
	#plot rectangles on the image
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
	cv2.imshow("Face Detector",frame)
	key = cv2.waitKey(1)
	# Quit if "q" is pressed
	if key == ord('q'):
		break
	# Save file if "w" is pressed
	if key == ord('w'):
		cv2.imwrite("face_detector.png",frame)

cap.release()
cv2.destroyAllWindows()