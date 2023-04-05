import cv2
import numpy as np
import pandas as pd


def ResizeImage(img, x1, y1, x2, y2, new_width=416, new_height=416):
    # Determine the original image dimensions
    original_height, original_width = img.shape[:2]

    # Calculate the scale factors
    width_scale = new_width / original_width
    height_scale = new_height / original_height

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate the corresponding points on the resized image
    x1_new, y1_new = int(x1 * width_scale), int(y1 * height_scale)
    x2_new, y2_new = int(x2 * width_scale), int(y2 * height_scale)

    # Draw the points on the resized image
    cv2.circle(img, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, (x2, y2), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.circle(resized_img, (x1_new, y1_new), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(resized_img, (x2_new, y2_new), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow('Resized Image', resized_img)
    cv2.imshow('Orignal Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized_img, x1_new, y1_new, x2_new, y2_new


def LoadDataSet(dataSetFolderPath: str):
    annotationsFilePath = dataSetFolderPath+"/allAnnotations.csv"
    annotationsDataFrame = pd.read_csv(annotationsFilePath, sep=";")
    imageSize = []
    for index, row in annotationsDataFrame[1:].iterrows():
        image = cv2.imread(dataSetFolderPath+"/"+row[0])
        imageSize.append([image.shape[0], image.shape[1]])
        resizedImage, resized_x1, resized_y1, resized_x2, resized_y2 = ResizeImage(
            image, row[2], row[3], row[4], row[5])
    imageSizeDF = pd.DataFrame(imageSize, columns=["width", "height"])
    uniqueSigns = annotationsDataFrame['Annotation tag'].unique()
    uniquewidth = imageSizeDF["width"].unique().tolist()
    uniqueheight = imageSizeDF["height"].unique().tolist()
    print(len(uniquewidth), uniquewidth)
    print(len(uniqueheight), uniqueheight)


LoadDataSet("./DataSet")
