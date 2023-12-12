import cv2
import os
import numpy as np
import fitz

outputdir="pdfimg"
white_lmt = 230
black_lmt = 180

def get_checkerboards_from_pdfimg(image, outimg):
    img = image.copy()
    h, w, c = img.shape
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(frame, 180, 230, cv2.THRESH_BINARY)
    erodeim = cv2.erode(mask, None, iterations=1)  # 腐蚀
    dilateim = cv2.dilate(erodeim, None, iterations=3)

    dst = cv2.bitwise_and(img, img, mask=dilateim)
    frame = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(frame, 180, 230, cv2.THRESH_BINARY)
    #contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    subarea_crit = h * w / 23
    checker_cont = [c for c in contours if h * w * 0.8 > cv2.contourArea(c) > subarea_crit]
    subcheckers = sorted(checker_cont, key=lambda c: cv2.contourArea(c), reverse = True)

    i = 0
    offset = 8
    #The biggest contour is the whole image. ??? so, it is ignored here.
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
        checkerboard = image[y1:y2, x1:x2]

        imgfile = f'{outimg}_p{i}.png'
        cv2.imwrite(imgfile, checkerboard)
        i+=1

def get_allcheckers_frompdf(filename, start, end=-1, step=1):
        if not os.path.exists(outputdir):
                os.makedirs(outputdir)

        pdf = fitz.open(filename)
        if end == -1:
                end = pdf.page_count
        pgnolist = list(range(start, end, step))
        trans = fitz.Matrix(1.5, 1.5)
        for pg in pgnolist:
                page = pdf[pg-1]
                pix = page.get_pixmap(matrix=trans, alpha=False)
                outimgprefix = os.path.join(outputdir, f"img{pg}")
                img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
                get_checkerboards_from_pdfimg(img, outimgprefix)
        pdf.close()
