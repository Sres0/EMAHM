import cv2 as cv
import numpy as np
import time

i = 52

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(i, hide=True):
    img = cv.imread(path(i))
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', resized)
    return resized

cap = createImg(i, hide=True)

#Capture frames from the camera
#Blur the image
blur = cv.blur(cap,(3,3))
#Convert to HSV color space
hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
frame = cap
#Create a binary image with where white will be skin colors 
#and rest is black
mask2 = cv.inRange(hsv,np.array([100,0,0]),np.array([255,165,255]))

#Kernel matrices for morphological transformation    
kernel_square = np.ones((11,11),np.uint8)
kernel_ellipse= cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

dilation = cv.dilate(mask2,kernel_ellipse,iterations = 1)
erosion = cv.erode(dilation,kernel_square,iterations = 1)    
dilation2 = cv.dilate(erosion,kernel_ellipse,iterations = 1)    
filtered = cv.medianBlur(dilation2,5)
kernel_ellipse= cv.getStructuringElement(cv.MORPH_ELLIPSE,(8,8))
dilation2 = cv.dilate(filtered,kernel_ellipse,iterations = 1)
kernel_ellipse= cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
dilation3 = cv.dilate(filtered,kernel_ellipse,iterations = 1)
median = cv.medianBlur(dilation3,5)
ret,thresh = cv.threshold(median,127,255,0)

#Find contours of the filtered frame
contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)   

#Draw Contours
cv.drawContours(cap, contours, -1, (122,122,0), 2)
#cv.imshow('Dilation',median)

#Find Max contour area (Assume that hand is in the frame)
max_area=100
ci=0	
for i in range(len(contours)):
    cnt=contours[i]
    area = cv.contourArea(cnt)
    if(area>max_area):
        max_area=area
        ci=i
        
# #Largest area contour	  
cnts = contours[ci]
#Draw Contours
# cv.drawContours(cap, cnts, -1, (122,122,0), 2)
#cv.imshow('Dilation',median)

# #Find convex hull
hull = cv.convexHull(cnts)

# #Find convex defects
hull2 = cv.convexHull(cnts,returnPoints = False)
defects = cv.convexityDefects(cnts,hull2)

# #Get defect points and draw them in the original image
FarDefect = []
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnts[s][0])
    end = tuple(cnts[e][0])
    far = tuple(cnts[f][0])
    FarDefect.append(far)
    # cv.line(cap,start,end,[0,255,0],1)
    cv.circle(cap,far,10,[100,255,255],3)

# #Find moments of the largest contour
moments = cv.moments(cnts)

# #Central mass of first order moments
if moments['m00']!=0:
    cx = int(moments['m10']/moments['m00']) # cx = M10/M00
    cy = int(moments['m01']/moments['m00']) # cy = M01/M00
centerMass=(cx,cy)    

# #Draw center mass
cv.circle(cap,centerMass,7,[100,0,255],2)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(cap,'Center',tuple(centerMass),font,2,(255,255,255),2)     

# #Distance from each finger defect(finger webbing) to the center mass
distanceBetweenDefectsToCenter = []
for i in range(0,len(FarDefect)):
    x =  np.array(FarDefect[i])
    centerMass = np.array(centerMass)
    distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
    distanceBetweenDefectsToCenter.append(distance)

# #Get an average of three shortest distances from finger webbing to center mass
sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

# #Get fingertip points from contour hull
# #If points are in proximity of 80 pixels, consider as a single point in the group
finger = []
for i in range(0,len(hull)-1):
    if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
        if hull[i][0][1] <500 :
            finger.append(hull[i][0])

# #The fingertip points are 5 hull points with largest y coordinates  
finger =  sorted(finger,key=lambda x: x[1])   
fingers = finger[0:5]

# #Calculate distance of each finger tip to the center mass
fingerDistance = []
for i in range(0,len(fingers)):
    distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
    fingerDistance.append(distance)

# #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
# #than the distance of average finger webbing to center mass by 130 pixels
result = 0
for i in range(0,len(fingers)):
    if fingerDistance[i] > AverageDefectDistance+0:
        result = result +1

# #Print number of pointed fingers
cv.putText(frame,str(result),(100,100),font,2,(255,255,255),2)

# #show height raised fingers
# #cv.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
# #cv.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
# #cv.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
# #cv.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
# #cv.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
# #cv.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
# #cv.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
# #cv.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
    
# #Print bounding rectangle
x,y,w,h = cv.boundingRect(cnts)
img = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow('img', cap)

cv.drawContours(frame,[hull],-1,(255,255,255),2)

# ##### Show final image ########
cv.imshow('Dilation',frame)

#close the output video by pressing 'ESC'
k = cv.waitKey(0) & 0xFF=='q'

# cap.release()
cv.destroyAllWindows()