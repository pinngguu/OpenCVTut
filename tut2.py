import cv2 as cv
import numpy as np



#________________CONTOURS_____________________
img = cv.imread('Resources/Photos/cats.jpg')

cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')
cv.waitKey(0)


"""
# ______________Transformations______________________
img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)

#Translation
def translate(img, x, y):
	transMat = np.float32([[1, 0, x],[0,1,y]])
	dimensions = (img.shape[1], img.shape[0])
	return cv.warpAffine(img, transMat, dimensions)

# [-x] --> Left
# [-y] --> Up
# [x] --> Right
# [y] --> Down

translated = translate(img, 100, 100)
cv.imshow('Translated', translated)

cv.waitKey(0)
"""

"""
# Reading images
img = cv.imread('Resources/Photos/cat_large.jpg')
cv.imshow('Cat', img)


# Reading Videos

def rescaleFrame(frame, scale = 0.75):
	
	# for Images, Videos and Live videos
	width = int(frame.shape[1] * scale)		#[1] denotes width, [0] for height
	height = int(frame.shape[0] * scale)

	dimensions = (width, height)

	return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
	# for Live video only
	capture.set(3, width)
	capture.set(4, height)


capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
	isTrue, frame = capture.read()
	frame_resized = rescaleFrame(frame, scale = 0.2)


	cv.imshow('Video', frame)
	cv.imshow('Video Resized', frame_resized)

	if cv.waitKey(20) & 0xFF==ord('d'):
		break

capture.release()
cv.destroyAllWindows()
"""