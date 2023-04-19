import cv2
import numpy as np
import pandas as pd
import imutils
import os

def LoadDataSet(dataSetFolderPath: str):
    annotationsFilePath = dataSetFolderPath+"/allAnnotations.csv"
    annotationsDataFrame = pd.read_csv(annotationsFilePath, sep=";")
    imageSize = []
    for index, row in annotationsDataFrame[1:].iterrows():
        image = cv2.imread(dataSetFolderPath+"/"+row[0])
        imageSize.append([image.shape[0], image.shape[1]])

    imageSizeDF = pd.DataFrame(imageSize, columns=["width", "height"])
    uniqueSigns = annotationsDataFrame['Annotation tag'].unique()
    uniquewidth = imageSizeDF["width"].unique().tolist()
    uniqueheight = imageSizeDF["height"].unique().tolist()
    print(len(uniquewidth), uniquewidth)
    print(len(uniqueheight), uniqueheight)


def SignDetection(image):
    try:
        grayScale = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    except:
        grayScale = Image
    bilateralFilter = cv2.bilateralFilter(grayScale, 11, 17, 17)
        
    #Calculating the upper ad lower thresolds for CannyEdge from the image
    imageMedian = np.median(image)
    lowerThreshold = max(0, (0.7 * imageMedian))
    upperThreshold = min(255, (0.7 * imageMedian))

    # Edge detection using cannyedge  
    cannyEdged = cv2.Canny(bilateralFilter, lowerThreshold , upperThreshold)
    (cnts, _) = cv2.findContours(cannyEdged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:]
    maxArea = 0
    SignsDetected = []
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True) 
        if (len(approx) == 4):  # Select the contour with 4 corners
            signArea = approx
            x,y,w,h = cv2.boundingRect(signArea)
            area = w*h
            if(area>=maxArea):
                maxArea = area
                if([x,y,w,h] not in SignsDetected):
                    SignsDetected.append([x,y,w,h])            
    return SignsDetected

allImageNames = os.listdir("./Traffic Signs")
for imageName in allImageNames:
    Image = cv2.imread("./Traffic Signs/"+imageName)
    SignsDetected = SignDetection(Image)
    i=1
    for [x,y,w,h] in SignsDetected:
        signBoard = Image[y:y+h , x:x+w]
        signBoard = imutils.resize(signBoard, width=500)
        if("(" in imageName and len(imageName.split('(')[0])!=0):
            imageName_ = imageName.split('(')[0]+ str(i)+".png"

        else:
            imageName_ = imageName.split(".")[0]+ str(i)+".png"
        cv2.imwrite("./TrafficSign DataSet/" + imageName_, signBoard)
        i+=1
    