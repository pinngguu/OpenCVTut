import numpy as np
import cv2

"""

#____________Bind Trackbar to OpenCV Window________________

def nothing(x):
    print(x)

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')


cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('R', 'image', 0, 255, nothing)

switch = '0 : OFF\n 1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    b = cv2.getTrackbarPos('B', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    r = cv2.getTrackbarPos('R', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]


cv2.destroyAllWindows()

"""


"""
#____________FACE AND EYE DETECTION__________________

cap = cv2.VideoCapture('assets/crush.mp4')

#modules for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detection by drawing a rectangle the object
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), 5)
        
        #grab Region of Interest (area) of face and eyes. It takes first y, then x
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+h]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 4)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex+ew, ey+eh), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""


"""
#__________________TEMPLATE DETECTION______________________

#import image and template, both need to resize and not just one.
img = cv2.resize(cv2.imread('assets/soccer_practice.jpg',0), (0,0), fx=0.5, fy=0.5)
template = cv2.resize(cv2.imread('assets/ball.PNG',0), (0,0), fx=0.5, fy=0.5)


h, w = template.shape

#all methods are called and will be used one by one.
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

#loop to see each method
for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method) 
    #dimensions are (W-w+1, H-h+1) 
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
      
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc  #Only these two methods give minimum location
    else:
        location = max_loc

    #find bottom right corner because location gives value of top left corner of surrounding rectangle
    bottom_right = (location[0] + w, location[1] + h)

    cv2.rectangle(img2, location, bottom_right, 255, 3)
    cv2.imshow('Match', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""



"""
#____________ CORNER DETECTION ________________ 

img = cv2.imread('assets/chessboard.png')
img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#manipulate the image in the grayscale and use it on the original image
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
      #[source file, no. of corners, min. quality(0 to 1), min. Euclidean distance b/w 2 corners

corners = np.int0(corners) #convert data into integers

for corner in corners:
	x, y = corner.ravel()          #flatten the image
	cv2.circle(img, (x, y), 5, (255, 0, 0), -1)


#draw lines between the dots
for i in range(len(corners)):
	for j in range(i + 1, len(corners)):           #cycle through the remaining corners
		corner1 = tuple(corners[i][0])             #convert corner values to tuple
		corner2 = tuple(corners[j][0])
        #get color value of the corner variable and convert into integer format of python using map function
		color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
		cv2.line(img, corner1, corner2, color, 1)   #draw the line

cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


""" 
# ________ CONVERT IMAGE TO HSV FROM BGR _____________

cap = cv2.VideoCapture(0) #Capture video using default camera (0)

while True:
    ret, frame = cap.read()
    #get height(4) and width(3) of image. Convert to int to slice properly.
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #____convert BGR to HSV and select a color range to extract
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([130,255,255])
    #____create a mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #____apply the mask on the image. BitwiseAnd compares two images and applies the mask
    result = cv2.bitwise_and(frame, frame, mask=mask)
        
    # Display the image until 'q' is pressed
    #cv2.imshow('Mask', mask)
    cv2.imshow('HSV Image', hsv)
    #cv2.imshow('colour in range', result)
    cv2.imshow('OG Image',frame)
    if cv2.waitKey(1) == ord('q'):
        break
# to release the hold of camera which is being used by OpenCV
cap.release()
cv2.destroyAllWindows() 
"""


""" 
#_______ DRAW LINES, SHAPES, TEXTS ON IMAGE ______________________

cap = cv2.VideoCapture(0) #Capture video using default camera (0)

while True:
    ret, frame = cap.read()
    #get height(4) and width(3) of image. Convert to int to slice properly.
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #draw the line, rectangle, circle, text on image
    imgg = cv2.line(frame, (0,0), (width, height), (255,128,255), 10)
                # [Source image, start posn, end posn, colour, thickness]
    imgg = cv2.line(imgg, (0,height), (width, 0), (255,255,128), 5)
                # [Source image, start posn, end posn, colour, thickness]
    imgg = cv2.rectangle(imgg, (100,100), (200,200), (128,128,255), -1)
                # [Source image, topLeft posn, bottomRight posn, colour, thickness]
    imgg = cv2.circle(imgg, (300,300), 60, (128,255,128), -1)
                # [Source image, centre posn, radius, colour, thickness]
    fontt = cv2.FONT_HERSHEY_SIMPLEX #Specify the font
    imgg = cv2.putText(imgg, 'Jannat', (200, height - 10), fontt, 2, (128,255,255), 5, cv2.LINE_AA)
                # [Source image, The Text, BottomRight corner, font, font scale, colour, thickness, line type(makes it look good)]
    
    # Display the image until 'q' is pressed
    cv2.imshow('frame',imgg)
    if cv2.waitKey(1) == ord('q'):
        break
# to release the hold of camera which is being used by OpenCV
cap.release()
cv2.destroyAllWindows() 
"""

""" 
#______GET LIVE FEED FROM WEBCAM AND DISPLAY IT IN FOUR QUADRANTS_______
 
cap = cv2.VideoCapture(0) #Capture video using default camera (0)

while True:
    ret, frame = cap.read()
    #get height(4) and width(3) of image. Convert to int to slice properly.
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Create a blank canvas of the same size as original image and shrink it
    image = np.zeros(frame.shape, np.uint8) 
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    #slice operation. Put image in each quadrant of empty canvas
    image[:height//2, :width//2] = smaller_frame #top left
    image[height//2:, :width//2] = cv2.rotate(smaller_frame, cv2.cv2.ROTATE_180) #bottom left
    image[:height//2, width//2:] = smaller_frame #top right
    image[height//2:, width//2:] = cv2.rotate(smaller_frame, cv2.cv2.ROTATE_180) #top right

    # Display the image until 'q' is pressed
    cv2.imshow('frame',image)
    if cv2.waitKey(1) == ord('q'):
        break
# to release the hold of camera which is being used by OpenCV
cap.release()
cv2.destroyAllWindows() 
 """

###----------------------------------------------------------------------------

""" 
#_______COPYING THE PIXELS______

img = cv2.imread("F:\pIYUSH\OpenCV_Python\Assets\deer.jpg",-1)

x = img[500:900, 600:900]
img[90:490, 150:450] = x
cv2.imshow('Image', cv2.resize(img, (768,432)))
cv2.waitKey(0)
cv2.destroyAllWindows() 
"""

