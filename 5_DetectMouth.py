import cv2
import matplotlib.pyplot as plt


# Load classifier
mouth_classifier = cv2.CascadeClassifier("./files/haarcascade_mcs_mouth.xml")

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
	mouth_rects = mouth_classifier.detectMultiScale(gray_frame,1.7,11)
	#plot rectangles on the image
	for (x,y,w,h) in mouth_rects:
		y=int(y-0.15*h)
		cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),3)
		break
	cv2.imshow("Mouth Detector",frame)
	key = cv2.waitKey(1)
	# Quit if "q" is pressed
	if key == ord('q'):
		break
	# Save file if "w" is pressed
	if key == ord('w'):
		cv2.imwrite("mouth_detector.png",frame)

cap.release()
cv2.destroyAllWindows()