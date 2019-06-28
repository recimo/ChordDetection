import cv2
import numpy as np
import math
from scipy.signal import argrelextrema

img = cv2.imread('a14.jpg')

resized = cv2.resize(img, (512, 512), cv2.INTER_LINEAR)

hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
lower_skin = np.array([0,55,95])
upper_skin = np.array([10,175,255]) #namestaj H parametar kako bi izdvojio boju koze bolje

skinMask = cv2.inRange(hsv_img, lower_skin, upper_skin)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #kernel za eroziju
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
skinMask = cv2.erode(skinMask, kernel2, iterations = 2) #dodatno erodirati za manje piksela (kernel2) zbog zica

skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skinMask = cv2.medianBlur(skinMask, 19)
skin = cv2.bitwise_and(hsv_img, hsv_img, mask=skinMask)

image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

ret,thresh = cv2.threshold(skinMask, 0, 255, 0)
im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    cv2.drawContours(image, contours, -1, 255, 3)
    #nadji najvecu konturu
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

hull = cv2.convexHull(c,returnPoints = False)
defects = cv2.convexityDefects(c,hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(c[s][0])
    end = tuple(c[e][0])
    far = tuple(c[f][0])
    #cv2.line(image,start,end,[0,255,0],2)
    cv2.circle(image,end,5,[0,0,255],-1)

#cv2.imshow('image', np.hstack([resized, image]))
#cv2.waitKey(0)
#cv2.destroyAllWindows()


################ trazenje linija (zica) gitare ################
#resized = cv2.medianBlur(resized, 11)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 25, 150)

kernelLine = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilatation = cv2.dilate(edges, kernelLine, iterations = 2)
erodion = cv2.erode(dilatation, kernelLine, iterations= 2)

lines = cv2.HoughLinesP(erodion, 1, np.pi/180, 50, maxLineGap=100, minLineLength=20)

horizontalLines = []
verticalLines = []
#iscrtavanje horizontalnih linija
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = math.atan2(y1 - y2, x1 - x2) ##################################
    angle = angle * 180 / np.pi          ## trazenje ugla za iscrtavanje
    if angle < 185 and angle > 175:      ##################################
        #cv2.line(resized, (x1,y1), (x2,y2), (0,255,0), 3)
        horizontalLines.append([(x1,y1),(x2,y2)]) #lista horizontalnih linija

### ***grupisi horizontalne*** ###
horizontalLinesTemp = horizontalLines #kopirana originalna lista
avgYtemp = []

for hl in horizontalLinesTemp:
    tempY = (hl[0][1] + hl[1][1]) / 2
    avgYtemp.append(tempY)

avgYtemp2 = avgYtemp
strings = []
similar = []

while len(avgYtemp) > 0:
    k = 0
    while len(avgYtemp2) > k:
        if abs(avgYtemp[0] - avgYtemp2[k]) < 20:
            similar.append(avgYtemp2[k])
            k += 1
        else:
            k += 1

    superY = sum(similar) / len(similar)
    strings.append(superY)
    for s in similar:
        avgYtemp.remove(s)

    similar.clear()

strings.sort()

for s in strings:
    y = int(round(s))
    cv2.line(resized, (0, y), (512, y), (0, 255, 0), 3)

### nadji i grupisi vertikalne ###
### nacrtaj vertikalne ###

lines = cv2.HoughLinesP(erodion, 1, np.pi/180, 50, maxLineGap=30, minLineLength=60)

verticalLines = []
#iscrtavanje horizontalnih linija
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = math.atan2(y1 - y2, x1 - x2) ##################################
    angle = angle * 180 / np.pi          ## trazenje ugla za iscrtavanje
    if angle < 95 and angle > 85:      ##################################
        if y2 < 256:
            cv2.line(resized, (x1,y1), (x2,y2), (0,255,0), 3)
            verticalLines.append([(x1,y1),(x2,y2)]) #lista vertikalnih linija

###grupisi vertikalne###

print(verticalLines)

verticalLinesTemp = verticalLines
arrayX = []
arrayX2 = []

for vl in verticalLines:
    avgX = (vl[0][0] + vl[1][0]) / 2
    arrayX.append(avgX)

arrayX.sort()
meanX = arrayX[0]

for x in arrayX:
    if abs(meanX - x) < 20:
        meanX = (meanX + x) / 2

##prvi sledeci prag##
for xx in arrayX:
    if xx > (meanX + 50):
        arrayX2.append(xx)

meanX2 = arrayX2[0]
for xx in arrayX2:
    if abs(meanX2 - xx) < 20:
        meanX2 = (meanX2 + xx) / 2

cv2.line(resized, (int(meanX),0), (int(meanX),512), (255,0,0), 3)

############################################################### trazenje svih pragova

fret1 = 0
fret2 = 0
fret3 = 0
fret4 = 0
ima3 = False
ima4 = False


if meanX < 100:
    fret4 = meanX
    #da li je treci ili cetvrt
    dif = abs(fret4 - meanX2)
    if(dif < 80):
        meanX2 = meanX2 + 40
        dif = abs(meanX2 - meanX)
    elif (dif > 150) and (dif < 250):
        meanX2 = (abs(meanX2-meanX) / 2) + meanX
        dif = abs(meanX2 - meanX)
    restPhoto = 512 - fret4
    if (restPhoto // dif) == 3:
        fret3 = fret4
        ima3 = True
    elif (restPhoto // dif) == 4 or (restPhoto // dif) == 5:
        fret3 = meanX2
        ima4 = True

    if ima4 == True:
        fret2 = fret3 + dif+5
        fret1 = fret2 + dif+10
    elif ima3 == True:
        fret2 = fret3 + dif+5
        fret1 = fret2 + dif+10

if ima3 == True:
    cv2.line(resized, (int(fret3), 0), (int(fret3), 512), (255, 0, 0), 3)
    cv2.line(resized, (int(fret2), 0), (int(fret2), 512), (255, 0, 0), 3)
    cv2.line(resized, (int(fret1), 0), (int(fret1), 512), (255, 0, 0), 3)

if ima4 == True:
    cv2.line(resized, (int(fret4), 0), (int(fret4), 512), (255, 0, 0), 3)
    cv2.line(resized, (int(fret3), 0), (int(fret3), 512), (255, 0, 0), 3)
    cv2.line(resized, (int(fret2), 0), (int(fret2), 512), (255, 0, 0), 3)
    cv2.line(resized, (int(fret1), 0), (int(fret1), 512), (255, 0, 0), 3)


cv2.imshow("edges", edges)
cv2.imshow("image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()



