import cv2
from cv2 import compare
import pytesseract
import numpy as np
import re
import random
from os import listdir
from os.path import isfile, join
from pytesseract import Output

def imgShow(source_photos, compare_photos):
    counter = 0
    for photo in source_photos:
        cv2.imshow("{}".format(random.randint(1, 1000)), source_photos[counter]["img_obj"])
        # cv2.waitKey(0)
        counter += 1
    counter = 0
    for photo2 in compare_photos:
        cv2.imshow("{}".format(random.randint(1, 1000)), compare_photos[counter]["img_obj"])
        # cv2.waitKey(0)
        counter += 1


def ImageToData(img_address, is_table):
    img_data = None
    if(is_table):
        img = cv2.imread(img_address)
        result = OCR_correct(img)
        img_data = pytesseract.image_to_data(result, output_type=Output.DICT, config='tessedit_char_whitelist=0123456789')
        img_data["img_obj"] = img
    else:
        img = cv2.imread(img_address)
        img_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        img_data["img_obj"] = img
        print(img_data)
    return img_data


def OCR_correct(img):
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # finding the background color for blending the lines for tesseract to identify it
    hist = cv2.calcHist([result], [0], None, [256], [0, 256])
    heighestIn = np.where(np.amax(hist) == hist)
    blendingColor = heighestIn[0][0].item()
    threshold = cv2.threshold(
        result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    removeHoriontal = cv2.morphologyEx(
        threshold, cv2.MORPH_OPEN, horizontalKernel, iterations=2)
    contursH = cv2.findContours(
        removeHoriontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contursH = contursH[0] if len(contursH) == 2 else contursH[1]
    for c in contursH:
        cv2.drawContours(
            result, [c], -1, (blendingColor, blendingColor, blendingColor), 5)

    VerticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    removeVertical = cv2.morphologyEx(
        threshold, cv2.MORPH_OPEN, VerticalKernel, iterations=2)
    contursV = cv2.findContours(
        removeVertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contursV = contursV[0] if len(contursV) == 2 else contursV[1]
    for c in contursV:
        cv2.drawContours(
            result, [c], -1, (blendingColor, blendingColor, blendingColor), 5)
    return result


def OCR_wtextCorrect(img):
    img = OCR_correct(img)
    text_mask = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    result = cv2.bitwise_and(img, img, mask=text_mask)
    return result


# 1) iterate over source pics
# 2) choose a source pic
# 3) iterate over source pic texts
# 4) extract it's data and iterate over compare pics
# 5) extract comapre picture data
# 6) a) compare source picture text with the compare pic text
    # *) if there is a match:
    # highlight both texts and up the counter of the compare pics array by one
    # **) if not and there is an another pic:
    # move to the other pic and try (a*) again
# 7) move to the next source pic (if there is one) and try 2 through 6 again

def MarkWords(img_sourceList, img_compareList):
    alpha = 0.5
    for s_imgData in img_sourceList:
        dic = s_imgData
        for number in range(len(dic["text"])):
            if (number != '' and int(float(dic['conf'][number])) > 60):
                bgr_markingColor = [random.randint(1,255),random.randint(1,255),random.randint(1,255)]
                overlaySource = dic["img_obj"].copy()
                # counter for going over the cmp pictures Data
                c_counter = 0
                # boolean if the number was found in both images
                num_found = False
                while (not num_found and c_counter < len(img_compareList)):
                    compare_dic = img_compareList[c_counter]
                    overlayComp = compare_dic["img_obj"].copy()
                    try:
                        # gives me the index of the matching text in the specific compare dictionary
                        # this line acts as a conditional, if there is no matches there will be a ValueError exception
                        cmpText_index = compare_dic["text"].index(dic["text"][number])

                        # highlighting the text on the source picture
                        width = dic["width"][number]
                        height = dic["height"][number]
                        xtop, ytop, xbottom, ybottom = dic["left"][number], dic["top"][
                            number], dic["left"][number] + width, dic["top"][number] + height
                        startingP = (int(xtop), int(ytop))
                        endP = (int(xbottom), int(ybottom))
                        cv2.rectangle(overlaySource, startingP,
                                      endP, (bgr_markingColor[0], bgr_markingColor[1], bgr_markingColor[2]), -1)
                        dic["img_obj"] = cv2.addWeighted(
                            overlaySource, alpha, dic["img_obj"], 1 - alpha, 0)

                        # highlighting the text on the compare picture
                        width = compare_dic["width"][cmpText_index]
                        height = compare_dic["height"][cmpText_index]
                        xtop, ytop, xbottom, ybottom = compare_dic["left"][cmpText_index], compare_dic["top"][
                            cmpText_index], compare_dic["left"][cmpText_index] + width, compare_dic["top"][cmpText_index] + height
                        startingP = (int(xtop), int(ytop))
                        endP = (int(xbottom), int(ybottom))
                        cv2.rectangle(overlayComp, startingP,
                                      endP, (bgr_markingColor[0], bgr_markingColor[1], bgr_markingColor[2]), -1)
                        compare_dic["img_obj"] = cv2.addWeighted(
                            overlayComp, alpha, compare_dic["img_obj"], 1 - alpha, 0)
                        num_found = True
                    except ValueError:
                        # up the counter by one and just move on to the next cmp pic
                        c_counter += 1

# find out how to read 2 images at the same time because i have trouble with that


# img = cv2.imread('C:\\tesseract\\bigPicture.jpeg')
# img2 = cv2.imread('C:\\tesseract\\green2.png')
# img3 = cv2.imread('C:\\tesseract\\greentable3.png')
# img4 = cv2.imread('C:\\tesseract\\sourceta.png')
img_sourceList = []
img_compareList = []
# result = OCR_correct(img)
# result2 = OCR_correct(img2)
# result3 = OCR_correct(img3)
# result4 = OCR_correct(img4)
# text = pytesseract.image_to_string(result)
# compNums = re.sub(" +", " ", re.sub("[^0-9]", " ", text)).split()
# img_data = pytesseract.image_to_data(result, output_type=Output.DICT)
# img2_data = pytesseract.image_to_data(result2, output_type=Output.DICT)
# img3_data = pytesseract.image_to_data(result3, output_type=Output.DICT)
# img4_data = pytesseract.image_to_data(result4, output_type=Output.DICT)
# img_data["img_obj"] = img
# img2_data["img_obj"] = img2
# img3_data["img_obj"] = img3
# img4_data["img_obj"] = img4
img_compareList.append(ImageToData('C:\\OCRProj\\bigPicture.jpeg', True))
img_sourceList.append(ImageToData('C:\\OCRProj\\green2.png', True))
img_sourceList.append(ImageToData('C:\\OCRProj\\greentable3.png', True))
img_sourceList.append(ImageToData('C:\\OCRProj\\sourceta.png', True))
img_sourceList.append(ImageToData('C:\\OCRProj\\whatsappImg.png', False))
MarkWords(img_sourceList, img_compareList)
# cv2.imshow("example", img_data["img_obj"])
# cv2.imshow("example2", img2_data["img_obj"])
# cv2.imshow("example3", img3_data["img_obj"])
# cv2.imshow("example4", img4_data["img_obj"])
imgShow(img_sourceList, img_compareList)
cv2.waitKey(0)

