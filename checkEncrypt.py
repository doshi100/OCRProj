import cv2
from cv2 import compare
import pytesseract
import numpy as np
import re
import random
from os import listdir
from os.path import isfile, join
from pytesseract import Output


def cropPaste(sourceList, compList):
    for dic in sourceList:
        cropped_img = dic["img_obj"]
        org_img = dic["org_img"]
        img_height = np.shape(cropped_img)[0]
        img_width = np.shape(cropped_img)[1]
        org_img[dic["crop_coor"][1]:img_height, dic["crop_coor"][0]:img_width] = dic["img_obj"]
        dic["org_img"] = org_img
    for dic in compList:
        cropped_img = dic["img_obj"]
        org_img = dic["org_img"]
        img_height = np.shape(cropped_img)[0] + dic["crop_coor"][0]
        img_width = np.shape(cropped_img)[1] + dic["crop_coor"][1]
        org_img[dic["crop_coor"][0]:img_height, dic["crop_coor"][1]:img_width] = dic["img_obj"]
        dic["org_img"] = org_img


# shows all the images that are in the compare and source img lists.
def imgShow(source_photos, compare_photos):
    counter = 0
    for photo in source_photos:
        cv2.imshow("{}".format(random.randint(1, 1000)), source_photos[counter]["org_img"])
        # cv2.waitKey(0)
        counter += 1
    counter = 0
    for photo2 in compare_photos:
        cv2.imshow("{}".format(random.randint(1, 1000)), compare_photos[counter]["org_img"])
        # cv2.waitKey(0)
        counter += 1

            #   (1) img location in memory
            #   (2) bool set to true if image depict a table
            #   (3) specify if the img was cropped and provide the original img
            #   (4) specify the box coordinates of the cropped image (tuple)
def ImageToData(img_address, is_table, orgImg=None, orgImg_coor=None):
    img_data = None
    if(is_table):
        img = cv2.imread(img_address)
        result = OCR_correct(img)
        img_data = pytesseract.image_to_data(result, output_type=Output.DICT, config='tessedit_char_whitelist=0123456789')
        img_data["img_obj"] = img
        if(orgImg is not None):
            org_img = cv2.imread(orgImg)
            img_data["org_img"] = org_img
            img_data["crop_coor"] = orgImg_coor
        # if the photo wasn't cropped, put the original image into the org_img as well as img_obj
        else:
            img_data["org_img"] = img
            img_data["crop_coor"] = (0,0)
    else:
        img = cv2.imread(img_address)
        img_data = pytesseract.image_to_data(img, output_type=Output.DICT, config=r'-l eng+heb')
        strip_char(img_data)
        img_data["img_obj"] = img
        if(orgImg is not None):
            img_data["org_img"] = orgImg
            img_data["crop_coor"] = orgImg_coor
        # if the photo wasn't cropped, put the original image into the org_img as well as img_obj
        else:
            img_data["org_img"] = img
            img_data["crop_coor"] = (0,0)
    return img_data


# method that strips any string from the numbers list that is not a number for filtering only the numbers from the list 
def strip_char(img_data):
    counter = 0
    for iden_num in img_data["text"]:
        img_data["text"][counter] = re.sub("[^0-9]", "", iden_num)
        counter += 1

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
            if (dic["text"][number] != '' and int(float(dic['conf'][number])) > 60):
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


img_sourceList = []
img_compareList = []
img_compareList.append(ImageToData('C:\\OCRProj\\bigPicture.jpeg', True, 'C:\\OCRProj\\nums.png', (0,259)))
img_sourceList.append(ImageToData('C:\\OCRProj\\green2.png', True))
img_sourceList.append(ImageToData('C:\\OCRProj\\greentable3.png', True))
img_sourceList.append(ImageToData('C:\\OCRProj\\sourceta.png', True))
img_sourceList.append(ImageToData('C:\\OCRProj\\whatsappImg.png', False))
img_sourceList.append(ImageToData('C:\\OCRProj\\whatsappImg2.png', False))
MarkWords(img_sourceList, img_compareList)
cropPaste(img_sourceList, img_compareList)
imgShow(img_sourceList, img_compareList)
# mybigimg = cv2.imread('C:\\OCRProj\\nums.png')
# mybigimg[0:675, 259:361] = img_compareList[0]["img_obj"]
# cv2.imshow("croppedimg", mybigimg)
cv2.waitKey(0)

