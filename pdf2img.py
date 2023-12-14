import cv2
import os
import numpy as np
import fitz
from pathlib import Path

outputdir="pdfimg"
white_lmt = 230
black_lmt = 180
white_color = (255, 255, 255)
black_color = (0, 0, 0)

# image contains one big or several checker boards in one whole pdf page
def get_checkerboards_from_img(image, outimg = None, outtxt = None):
    img = image.copy()
    img_txt = image.copy()
    h, w, c = img.shape
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(frame, black_lmt, white_lmt, cv2.THRESH_BINARY)
    erodeim = cv2.erode(mask, None, iterations=1)  # 腐蚀
    dilateim = cv2.dilate(erodeim, None, iterations=3)

    dst = cv2.bitwise_and(img, img, mask=dilateim)
    frame = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(frame, black_lmt, white_lmt, cv2.THRESH_BINARY)
    #contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #if it is too big or too small, it will not the invalid checkerboard
    subarea_crit = h * w / 23
    subcheckers = [c for c in contours if h * w * 0.85 > cv2.contourArea(c) > subarea_crit]

    i = 0
    offset = 8
    checkers = []
    for b in subcheckers:
        # 多边形拟合
        epsilon = 0.1 * cv2.arcLength(b, True)
        if epsilon < 1:
            print("error :   epsilon < 1")
            #pass

        # 多边形拟合
        approx = cv2.approxPolyDP(b, epsilon, True)
        edges = sorted(approx, key = lambda p: p[0][0] + p[0][1])
        [[x1, y1]] = edges[0]
        [[x2, y2]] = edges[-1]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1, x2 = x1 - offset, x2 + offset
        y1, y2 = y1 - offset, y2 + offset
        cv2.rectangle(img_txt, [x1, y1], [x2, y2], white_color, -1)
        checkers.append(np.array((x1, y1, x2, y2)))

    checkers = sorted(checkers, key = lambda p: p[1])
    for checker in checkers:
        if outimg:
            checkerboard = image[checker[1]:checker[3], checker[0]:checker[2]]
            imgfile = f'{outimg}_p{i}.png'
            cv2.imwrite(imgfile, checkerboard)
        i+=1

    if outtxt:
        cv2.imwrite(img_txt, outtxt)
    return checkers

def get_allcheckers_frompdf(filename, start, end=-1, step=1):
        if not os.path.exists(outputdir):
                os.makedirs(outputdir)

        pdf = fitz.open(filename)
        if end == -1:
                end = pdf.page_count

        ppi = 300
        scale = ppi / 70
        pgnolist = list(range(start, end, step))
        trans = fitz.Matrix(scale, scale)
        for pg in pgnolist:
                page = pdf[pg-1]
                pix = page.get_pixmap(matrix=trans, alpha=False)
                outimgprefix = os.path.join(outputdir, f"img{pg}")
                outtext = os.path.join(Path(filename).parent, f"{pg}-ocr.png")
                image = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
                get_checkerboards_from_img(image, outimgprefix, outtext)
        pdf.close()
