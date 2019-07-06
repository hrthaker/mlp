import cv2
import matplotlib.pyplot as plt


# Load classifier
nose_classifier = cv2.CascadeClassifier("./files/haarcascade_mcs_nose.xml")

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
	nose_rects = nose_classifier.detectMultiScale(gray_frame,1.7,11)
	#plot rectangles on the image
	for (x,y,w,h) in nose_rects:
		cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),3)
		break
	cv2.imshow("Nose Detector",frame)
	key = cv2.waitKey(1)
	# Quit if "q" is pressed
	if key == ord('q'):
		break
	# Save file if "w" is pressed
	if key == ord('w'):
		cv2.imwrite("nose_detector.png",frame)

cap.release()
cv2.destroyAllWindows()