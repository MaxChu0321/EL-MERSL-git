# 適用於從 LCAA (Liver Cancer Auto Annotation) 軟體產出的 case result 分析
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt

import largestinteriorrectangle as lir


# Contour 系列
def selectTumorContour(results_dir: Path, max_k:int =1) -> Tuple[int, np.array]:
    ''' 從 case result 資料夾中，分析所有 labeled 影像找出最大 contour
    Args:
        results_dir: The path of case result
    Returns:
        max_idx: the index of the slice with max contour
        max_contour: the max contour
    '''
    results_pth = results_dir.glob("./*.png")

    # Get contours
    results = dict()
    for r_pth in results_pth:
        r_idx = int(r_pth.stem)
        # print(f"Parse result: {r_idx}...")
        output = Image.open(r_pth)
        arr = np.array(output)
        contours = getContours(arr)
        results[r_idx] = contours

    # Get max contours
    max_contour = None
    max_idx = -1
    max_pixel_area = 0
    for idx, ctrs in results.items():
        if not ctrs: continue
        for ctr in ctrs:
            area = cv2.contourArea(ctr)
            if area > max_pixel_area:
                max_contour = ctr
                max_idx = idx
                max_pixel_area = area

    return max_idx, max_contour


def getContours(img: np.array) -> list:
    ''' 找出 contour (因為 contour 有厚度所以有分外圈內圈) -> 取外圈
    Ａrgs:
        img: grayscaled image array, default img size: 512*512
    Returns:
        out_contours: contours
    '''
    img = onlyContours(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    out_contours = [ contours[h[3]] for h in hierarchy[0] if h[3] != -1] # the fourth element is child-> parent index
    return out_contours


def onlyContours(img: np.array) -> np.array:
    '''把label的部分做二值化，輸出只有label的黑白影像
    Args:
        img: original image
    Returns:
        bin: binary image
    '''
    rows, cols, els = img.shape
    bin = np.zeros((rows, cols), dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            if (img[i,j,0] == img[i,j,1] == img[i,j,2]):
                bin[i,j] = 0
            else: bin[i,j] = 255

    return bin


def getMask(contour: np.array, shape: tuple) -> np.array:
    ''' 取得 contour roi 遮罩
    Args:
        contour:
        shape: the size of original image
    Returns:
        mask: roi 遮罩(1->True, 0->False)
    '''
    mask = np.zeros(shape, dtype='uint8')
    mask = cv2.drawContours(mask, [contour], 0, 1, cv2.FILLED)
    return mask


def getRectMask(contour: np.array, shape: tuple, img: np.array=None) -> np.array:
    ''' 取得 contour 內接矩形遮罩
    Args:
        contour:
        shape: the size of original image
        img: original grayscale image->if exists, show result
    Returns:
        mask: roi 遮罩(1->True, 0->False)
    '''
    roi = getMask(contour, shape).astype('bool')
    rectangle = lir.lir(roi)

    mask = np.zeros(shape, dtype='uint8')
    cv2.rectangle(mask, lir.pt1(rectangle), lir.pt2(rectangle), 1, cv2.FILLED)

    if type(img) != None:
        x, y, w, h = rectangle
        test_img = img.copy()
        test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        test_img = cv2.circle(test_img, lir.pt1(rectangle),0, (255,0,0), 5)
        test_img = cv2.circle(test_img, lir.pt2(rectangle),0, (255,0,0), 5)
        cv2.rectangle(test_img, lir.pt1(rectangle), lir.pt2(rectangle), (0,255,0), 1)
        cv2.drawContours(test_img, [contour],0,(0,0,255),1)
        plt.imshow(test_img, cmap="gray")
        plt.show()
    
    return mask


def cropInterRect(img: np.array, contour: tuple) -> np.array:
    ''' 取得 contour 內接矩形 roi
    Args:
        img:
        contour:
    Returns:
        cropped_img: 
    '''
    roi = getMask(contour, img.shape).astype('bool')
    x, y, w, h = lir.lir(roi)

    return img[y:y+h, x:x+w]


# Edge 系列
def getEdgeMasks(contour: np.array, shape: tuple, thickness: int=5) -> Tuple[np.array, np.array]:
    ''' 取得 contour 向內與向外兩遮罩
    Args:
        contour:
        shape: the size of original image
        thickness:
    Returns:
        ctr_mask: 沿著 contour, thickness*2 的遮罩包含in_mask和out_mask(1->True, 0->False)
        in_mask: 向內的遮罩 (1->True, 0->False)
        out_mask: 向外的遮罩(1->True, 0->False)
    '''
    ctr_val = 50
    in_val  = 100
    in_threshold = ctr_val+in_val
    out_threshold = ctr_val
    # Draw contour with thickness
    ctr_mask = np.zeros(shape, dtype='uint8')
    ctr_mask = cv2.drawContours(ctr_mask, [contour], 0, ctr_val, thickness*2)

    # Draw contour with filled region
    ctr_region = np.zeros(shape, dtype='uint8')
    ctr_region = cv2.drawContours(ctr_region, [contour], 0, in_val, cv2.FILLED)

    # Overlapped
    overlapped = ctr_mask+ctr_region
    in_mask = np.where(overlapped == in_threshold, 1, 0).astype(np.uint8)
    out_mask = np.where(overlapped == out_threshold, 1, 0).astype(np.uint8)

    ctr_mask = np.where(ctr_mask == ctr_val, 1, 0).astype(np.uint8)
    
    return ctr_mask, in_mask, out_mask