# Submitted by 
# Anudeep Kalitkar
# Parikshit Rohit Pravin Srivastav
from copy import deepcopy
from random import randint
import numpy as np
import math
from PIL import Image
from PIL.ImageShow import show

def SaveImage(name: str, image: np, displayImage: bool = False) -> None:
    im = Image.fromarray(image.astype(np.uint8))
    im.save("./output Images/"+name+".png")
    if(displayImage):
        show(im, name)

def GrayScaleConverstion(image: np, greyScalefilter: np =np.array([0.2989, 0.5870, 0.1140])) -> np:
    imagecopy = deepcopy(image)
    greyScaleImage = np.dot(imagecopy[:,:,:3], greyScalefilter)
    return greyScaleImage


def AddGaussianNoise(image: np,mean: float, std : float):
    noise = np.random.normal(mean, std, size = image.shape)
    noisyImage = image + noise
    return noisyImage 

def AddSaltPepperNoise(image: np, percent: int) -> np:
    saltPepperImage = np.copy(image)
    saltPepperCount = (((image.shape[0]*image.shape[1])//100)*percent)//2
    for i in range(saltPepperCount):
        saltPepperPixel = [randint(0,image.shape[0])-1, randint(0,image.shape[1] )-1]
        saltPepperImage[saltPepperPixel[0], saltPepperPixel[1]] = 0 
        saltPepperPixel = [randint(0,image.shape[0])-1, randint(0,image.shape[1] )-1]
        saltPepperImage[saltPepperPixel[0], saltPepperPixel[1]] = 255
    return saltPepperImage 

def MedianFilter(image: np, k: int) -> np:
    outputImage = np.zeros(image.shape)
    padding = (k//2)*2
    paddedImage = np.zeros((image.shape[0]+padding, image.shape[1]+padding))
    # print(image.shape, paddedImage.shape)
    paddedImage[(padding//2):paddedImage.shape[0] - (padding//2), (padding//2):paddedImage.shape[1] - (padding//2)] = image
    noRows, noCols = image.shape
    for i in range(noRows):
        for j in range(noCols):
            try:
                outputImage[i,j] = np.median(paddedImage[i:i+k,j:j+k])
            except:
                print("error")
                # outputImage[i,j] = image[i,j]
    return outputImage

def FindNorm(k: float, std: int, mean: int = 0):
    denominator = (math.sqrt(2* math.pi)*std)
    expo = -((k**2)/ 2*(std**2))
    return (math.e**expo)/denominator

def GenerateGaussianKernel(size: int, std: int = 1) -> np:
    OneDKernel = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        OneDKernel[i] = FindNorm(OneDKernel[i], std)
    twoDKernel = np.outer(OneDKernel, OneDKernel)
    twoDKernel *= 1.0 / twoDKernel.max()
    return twoDKernel

def GenerateBoxKernel(size: int) -> np:
    boxFilter = np.zeros((size,size))
    boxFilter += 1
    boxFilter = boxFilter /np.sum(boxFilter)
    return boxFilter 


def Convolve(image: np, kernel: np) -> np:
    outputImage = np.zeros(image.shape)
    noRows, noCols = image.shape
    noKRows, noKCols = kernel.shape
    paddingHeight, paddingWidth  = (noKRows//2)*2 , (noKCols//2)*2
    paddedImage = np.zeros((noRows + paddingHeight, noCols + paddingWidth))
    paddedImage[(paddingHeight//2):paddedImage.shape[0] - (paddingHeight//2), (paddingWidth//2):paddedImage.shape[1] - (paddingWidth//2)] = image
    for i in range(noRows):
        for j in range(noCols):
            outputImage[i,j] = np.sum(kernel*paddedImage[i:i+noKRows, j:j+noKCols])
    return outputImage

# question1
image = np.array(Image.open('8bit image.png'))
greyScaleFilter = np.array([0.2989, 0.5870, 0.1140])
greyScaleImage = GrayScaleConverstion(image, greyScaleFilter)
SaveImage("greyScaleImage", greyScaleImage)

# question2 
mean = 0
gaussianNoisyImage1 = AddGaussianNoise(greyScaleImage,mean,1)
SaveImage("gaussianNoisyImage1", gaussianNoisyImage1)

gaussianNoisyImage10 = AddGaussianNoise(greyScaleImage,mean,10)
SaveImage("gaussianNoisyImage10", gaussianNoisyImage10)

gaussianNoisyImage30 = AddGaussianNoise(greyScaleImage,mean,30)
SaveImage("gaussianNoisyImage30", gaussianNoisyImage30)

gaussianNoisyImage50 = AddGaussianNoise(greyScaleImage,mean,50)
SaveImage("gaussianNoisyImage50", gaussianNoisyImage50)

# question 3
saltPepperNoisyImage10 = AddSaltPepperNoise(greyScaleImage,10)
SaveImage("saltPepperNoisyImage10", saltPepperNoisyImage10)

saltPepperNoisyImage30 = AddSaltPepperNoise(greyScaleImage,30)
SaveImage("saltPepperNoisyImage30", saltPepperNoisyImage30)

# question 4
boxKernel3x3 = GenerateBoxKernel(3)
boxFilter3x3ImageOnGaussianNoisyImage50 = Convolve(gaussianNoisyImage50, boxKernel3x3)
SaveImage("boxFilter3x3ImageOnGaussianNoisyImage50", boxFilter3x3ImageOnGaussianNoisyImage50)
boxFilter3x3ImageOnsaltPepperNoisyImage30 = Convolve(saltPepperNoisyImage30, boxKernel3x3)
SaveImage("boxFilter3x3ImageOnsaltPepperNoisyImage30", boxFilter3x3ImageOnsaltPepperNoisyImage30)


median3x3FilterImageOnGaussianNoisyImage50= MedianFilter(gaussianNoisyImage50,3)
SaveImage("median3x3FilterImageOnGaussianNoisyImage50", median3x3FilterImageOnGaussianNoisyImage50)
median3x3FilterImageOnSaltPepperNoisyImage30 = MedianFilter(saltPepperNoisyImage30,3)
SaveImage("median3x3FilterImageOnSaltPepperNoisyImage30", median3x3FilterImageOnSaltPepperNoisyImage30)


gaussianKernel3x3 = GenerateGaussianKernel(3,5)
gaussian3x3FilterImageOnGaussianNoisy50 = Convolve(gaussianNoisyImage50,gaussianKernel3x3)
SaveImage("gaussian3x3FilterImageOnGaussianNoisy50", gaussian3x3FilterImageOnGaussianNoisy50)
gaussian3x3FilterImageOnSaltPepperNoisyImage30  = Convolve(saltPepperNoisyImage30,gaussianKernel3x3)
SaveImage("gaussian3x3FilterImageOnSaltPepperNoisyImage30", gaussian3x3FilterImageOnSaltPepperNoisyImage30)


#Question 5
boxKernel5x5 = GenerateBoxKernel(5)
boxFilter5x5ImageOnGaussianNoisyImage50 = Convolve(gaussianNoisyImage50, boxKernel5x5)
SaveImage("boxFilter5x5ImageOnGaussianNoisyImage50", boxFilter5x5ImageOnGaussianNoisyImage50)
boxFilter5x5ImageOnsaltPepperNoisyImage30 = Convolve(saltPepperNoisyImage30, boxKernel5x5)
SaveImage("boxFilter5x5ImageOnsaltPepperNoisyImage30", boxFilter5x5ImageOnsaltPepperNoisyImage30)


median5x5FilterImageOnGaussianNoisyImage50= MedianFilter(gaussianNoisyImage50,5)
SaveImage("median5x5FilterImageOnGaussianNoisyImage50", median5x5FilterImageOnGaussianNoisyImage50)
median5x5FilterImageOnSaltPepperNoisyImage30 = MedianFilter(saltPepperNoisyImage30,5)
SaveImage("median5x5FilterImageOnSaltPepperNoisyImage30", median5x5FilterImageOnSaltPepperNoisyImage30)

gaussianKernel5x5 = GenerateGaussianKernel(5,3)
print(gaussianKernel5x5)
gaussian5x5FilterImageOnGaussianNoisy50 = Convolve(gaussianNoisyImage50,gaussianKernel5x5)
SaveImage("gaussian5x5FilterImageOnGaussianNoisy50", gaussian5x5FilterImageOnGaussianNoisy50)
gaussian5x5FilterImageOnSaltPepperNoisyImage30  = Convolve(saltPepperNoisyImage30,gaussianKernel5x5)
SaveImage("gaussian5x5FilterImageOnSaltPepperNoisyImage30", gaussian5x5FilterImageOnSaltPepperNoisyImage30)

boxKernel9x9 = GenerateBoxKernel(9)
boxFilter9x9ImageOnGaussianNoisyImage50 = Convolve(gaussianNoisyImage50, boxKernel9x9)
SaveImage("boxFilter9x9ImageOnGaussianNoisyImage50", boxFilter9x9ImageOnGaussianNoisyImage50)
boxFilter9x9ImageOnsaltPepperNoisyImage30 = Convolve(saltPepperNoisyImage30, boxKernel9x9)
SaveImage("boxFilter9x9ImageOnsaltPepperNoisyImage30", boxFilter9x9ImageOnsaltPepperNoisyImage30)

median9x9FilterImageOnGaussianNoisyImage50= MedianFilter(gaussianNoisyImage50,9)
SaveImage("median9x9FilterImageOnGaussianNoisyImage50", median9x9FilterImageOnGaussianNoisyImage50)
median9x9FilterImageOnSaltPepperNoisyImage30 = MedianFilter(saltPepperNoisyImage30,9)
SaveImage("median9x9FilterImageOnSaltPepperNoisyImage30", median9x9FilterImageOnSaltPepperNoisyImage30)

gaussianKernel9x9 = GenerateGaussianKernel(9,5)
gaussian9x9FilterImageOnGaussianNoisy50 = Convolve(gaussianNoisyImage50,gaussianKernel9x9)
SaveImage("gaussian9x9FilterImageOnGaussianNoisy50", gaussian9x9FilterImageOnGaussianNoisy50)
gaussian9x9FilterImageOnSaltPepperNoisyImage30  = Convolve(saltPepperNoisyImage30,gaussianKernel9x9)
SaveImage("gaussian9x9FilterImageOnSaltPepperNoisyImage30", gaussian9x9FilterImageOnSaltPepperNoisyImage30)


