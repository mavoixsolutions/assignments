from cv2 import cv2
import sys
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ocr_simple.py image.jpg')
        sys.exit(1)
    # Read image path from command line
    imPath = sys.argv[1]
       
    # Uncomment the line below to provide path to tesseract manually
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    config = ('-l eng --oem 1 --psm 11')
    # Read image from disk
    im = cv2.imread(imPath, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # im2 = cv2.filter2D(im2, -1, kernel)
    # im2 = unsharp_mask(im2)
    # cv2.imwrite("./grayscale/sharp_"+imPath[-8:],im2)
    (thresh, im2) = cv2.threshold(im2, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im2 = unsharp_mask(im2)
    im2 = 255- im2

    # ret,thresh_value = cv2.threshold(im2,180,255,cv2.THRESH_BINARY_INV)
    # kernel = np.ones((5,5),np.uint8)
    # dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)
    # contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cordinates = []
    # count=0
    # ROI=[None,None,None]
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cordinates.append((x,y,w,h))
    #     count+=1
    #     #bounding the images
    #     if y< 50 and cv2.contourArea(cnt)>3000:
    #         cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),5)
    #         ROI=im2[x:x+w,y:y+h]
    # ROI=unsharp_mask(ROI)
    # ret,thresh_value = cv2.threshold(ROI,180,255,cv2.THRESH_BINARY_INV)
    # dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)
    # contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cordinates.append((x,y,w,h))
    #     count+=1
    #     #bounding the images
    #     if y< 50 and cv2.contourArea(cnt)>500:
    #         cv2.rectangle(ROI,(x,y),(x+w,y+h),(0,0,255),5)
        
    # cv2.drawContours(ROI, contours, -1, (0, 255, 0), 3) 
    cv2.imwrite("detecttable.jpg",im2)
    # cv2.namedWindow(‘detecttable’, cv2.WINDOW_NORMAL)
    # cv2.imwrite("detecttable.jpg",ROI)   
    # Run tesseract OCR on image
    text = pytesseract.image_to_string(im2, config=config)
    # Print recognized text
    print(text)