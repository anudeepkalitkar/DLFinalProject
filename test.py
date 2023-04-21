import os
allfolderNames = os.listdir("./TrafficSign DataSet")
annotationFile = open("./tafficSignAnnotations.csv",'w')
allImages = []
for folderName in allfolderNames:
    try:
        allImageNames = os.listdir("./TrafficSign DataSet/"+folderName)
        for imageName in allImageNames:
            annotationFile.write(folderName+", /TrafficSign DataSet/"+folderName +"/"+imageName +"\n")
    except:
        pass

# annotationFile.writelines(allImages)
annotationFile.close()
