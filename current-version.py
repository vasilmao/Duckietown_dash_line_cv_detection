import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from math import sqrt
from math import exp
from math import log
import os
import sys
import multiprocessing
import random

MINPIXEL = 3
MAXPIXEL = 110
RATIORDIST = 0.5
HEAP_CONST = 2.5
PARTOFINTEREST = 0.7
EPS = 8
MINHC = 4.5
MAXHC = 3
MAX_CNT_IN_BIG_RADIUS = 4
MAXDIST_BETWEEN_ELEMENTS_IN_SEQUENT_FRAMES = 50 # сделать функцию от расстояния по экспоненте

MAXRECC = 0.8
MINRECC = 0.1

CAMERA_MATRIX = np.array(
    [[278.79547761007365, 0.0, 314.29374336264345], [0.0, 280.52395701002115, 228.59132685202135], [0.0, 0.0, 1.0]])
DISTORTION_COEFFFICIENTS = np.array(
    [-0.23670917420627122, 0.03455456424406622, 0.0037778674941860426, 0.0020245279929775382, 0.0])


class LineDetector:
    class Element:
        def __init__(self, ncenter, ndx, ndy):
            self.center = ncenter
            self.dx = int(ndx)
            self.dy = int(ndy)

    def __init__(self, use_probability_filter=True):
        self.lastdashline = []
        self.use_probability_filter = use_probability_filter

    def _get_focus(self, el):
        x = el.dx
        y = el.dy
        if (x > y):
            a = sqrt(x*x - y*y)
            ans = [[el.center[0] + a, el.center[1]], [el.center[0] - a, el.center[1]]]
            return ans

        a = sqrt(y*y - x*x)
        #print("kek", a)

        ans = [[el.center[0], el.center[1] + a], [el.center[0], el.center[1] - a]]
        return ans



    def _pix_in_element(self, pix, el):
        f = self._get_focus(el)
        point = [el.center[0] + el.dx, el.center[1]]
        dist = self._distPixels(f[0], point) + self._distPixels(f[1], point)
        disttocheck = self._distPixels(f[0], pix) + self._distPixels(f[1], pix)
        return disttocheck <= dist



    def _intercect(self, el1, el2):
        xmin = max(el1.center[0] - el1.dx, el2.center[0] - el2.dx)
        xmax = min(el1.center[0] + el1.dx, el2.center[0] + el2.dx)
        ymin = max(el1.center[1] - el1.dy, el2.center[1] - el2.dy)
        ymax = min(el1.center[1] + el1.dy, el2.center[1] + el2.dy)
        if not (xmin <= xmax and ymin <= ymax):
            return False
        point = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        if (self._pix_in_element(point, el1) or self._pix_in_element(point, el2)):
            return True
        return False


    def _pix_in_element(self, pix, el):
        indx = (el.center[0] - el.dx <= pix[0] <= el.center[0] + el.dx)
        indy = (el.center[1] - el.dy <= pix[1] <= el.center[1] + el.dy)
        return indx and indy

    def _printelement(self, img, element, color):
        cv2.ellipse(img, (int(element.center[0]), int(element.center[1])), (element.dx, element.dy), 0, 0, 360, color,
                    thickness=2)
        return img
        # cv2.ellipse(img, (int(ncent[0]), int(ncent[1])), (dx, dy), 0, 0, 360, (0, 0, 255), thickness=1)

    def _maxRadius(self, h, lenImg):
        if h > lenImg * PARTOFINTEREST:
            return 0
        alpha = (1 - (MINPIXEL + EPS) / MAXPIXEL) / (lenImg * PARTOFINTEREST)
        return MAXPIXEL * (1 - alpha * h)

    def _heap_near(self, h, lenimg):
        h = max(0, h - lenimg * (1 - PARTOFINTEREST))
        return max(0, MINHC * exp(log(MAXHC / MINHC) / (lenimg * PARTOFINTEREST) * h))

    def _recratio(self, h, lenimg):
        h = max(0, h - lenimg * (1 - PARTOFINTEREST))
        return max(0, MINRECC * exp(log(MAXRECC / MINRECC) / (lenimg * PARTOFINTEREST) * h))

    def _maxRadius2(self, h, lenImg):
        if h > lenImg * PARTOFINTEREST:
            return 0
        alpha = math.log(EPS / MAXPIXEL) / (lenImg * PARTOFINTEREST)
        return MAXPIXEL * math.exp(alpha * h) + MINPIXEL

    def _distPixels(self, pix1, pix2):
        # print(pix1, pix2)
        return int(sqrt((pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2))

    def _viewImage(self, image, name_of_window):
        cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
        cv2.imshow(name_of_window, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _filter_contours(self, contours, img):
        filtered_contours = []

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            center = sum(approx) / approx.shape[0]
            # cv2.circle(img, (int(center[0][0]), int(center[0][1])), 3, (0, 255, 0))

            if len(approx) == 1:
                continue

            radius = max(cnt, key=lambda x: abs(center[0][0] - x[0][0]) ** 2 + abs(center[0][1] - x[0][1]) ** 2)
            radius_length = ((radius[0][0] - center[0][0]) ** 2 + (radius[0][1] - center[0][1]) ** 2) ** (0.5)

            if not MINPIXEL < radius_length < MAXPIXEL:
                continue

            # cv2.circle(img, (int(center[0][0]), int(center[0][1])), int(radius_length), (0, 0, 255), thickness=1)
            # cv2.putText(img, str(len(approx)), (int(center[0][0]), int(center[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness= 1)

            if len(approx) == 4 or len(approx) == 2 or len(approx) == 3:
                # alpha = 1/(len(img)*PARTOFINTEREST)
                curmaxr = self._maxRadius(len(img) - center[0][1], len(img))

                # print(*approx)

                if MINPIXEL < radius_length < MAXPIXEL:
                    # cv2.circle(img, (int(center[0][0]), int(center[0][1])), int(radius_length), (255, 0, 0), thickness= 2)
                    # cv2.putText(img, str(len(approx)), (int(center[0][0]), int(center[0][1])), 1, 1, (0, 255, 0), thickness= 2)
                    pass

                dx = 0
                dy = 0
                for el in approx:
                    x1 = int(el[0][0])
                    y1 = int(el[0][1])
                    for el2 in approx:
                        x2 = int(el2[0][0])
                        y2 = int(el2[0][1])
                        dx = max(dx, abs(x2 - x1))
                        dy = max(dy, abs(y2 - y1))

                dy = max(dy, int(dx * 0.4))
                dx //= 2
                dy //= 2
                ans = []
                maxdist = 0
                for el in approx:
                    for el2 in approx:
                        if self._distPixels(el[0], el2[0]) > maxdist:
                            maxdist = self._distPixels(el[0], el2[0])
                            ans = [el[0], el2[0]]
                ncent = [(ans[0][0] + ans[1][0]) // 2, (ans[0][1] + ans[1][1]) // 2]
                # cv2.circle(img, (int(ncent[0]), int(ncent[1])) , 2,  (0, 0, 255), thickness = 2)
                # cv2.ellipse(img, (int(ncent[0]), int(ncent[1])), (dx, dy), 0, 0 , 360, (0, 0, 255), thickness = 1)

                if not MINPIXEL < radius_length < MAXPIXEL:
                    continue

                # for lastEl in lastDashline:
                #     if distPixels(lastEl[0], center[0]) < MAXDIST_BETWEEN_ELEMENTS_IN_SEQUENT_FRAMES:
                #         trueDashline.append(Element(ncent, dx, dy))
                #         break

                if not MINPIXEL < radius_length < curmaxr:
                    continue
                    pass

                if all(img[int(center[0][1]), int(center[0][0])] == [0, 0, 0]) and len(
                        approx) != 2:  # удаляем черную разметку
                    continue
                    pass

                filtered_contours.append(self.Element(ncent, dx, dy))

        return filtered_contours

    # def isIntersect(a, b):
    #     grRadius = int(heap_near(a[0][1], len(img)) * a[1])
    #     smRadius = int(grRadius * recratio(a[0][1], len(img)))
    #     if (a[0][0] - grRadius <= b[0][0] <= a[0][0] + grRadius) and (a[0][1] - smRadius <= b[0][1] <= a[0][1] + smRadius):
    #         return True
    #     return False

    def _drawNearRectangles(self, img, dashline):
        for a in dashline:
            grRadius = int(self._heap_near(a[0][1], len(img)) * a[1])
            cv2.circle(img, (int(a[0][0]), int(a[0][1])), int(grRadius), (255, 255, 0), thickness=2)
            smRadius = int(grRadius * self._recratio(a[0][1], len(img)))
            # cv2.circle(img, (int(a[0][0]), int(a[0][1])), int(smRadius), (255, 0, 0), thickness= 2)

            # print('yay ', a[0][0])
            startpoint = (int(a[0][0] - grRadius), int(a[0][1] - smRadius))
            endpoint = (int(a[0][0] + grRadius), int(a[0][1] + smRadius))
            cv2.rectangle(img, startpoint, endpoint, (0, 255, 0), thickness=1)
        return img

    def _help_delete_heaps(self, dashline, img, mask):
        # img = drawNearRectangles(img, dashline)
        n = len(dashline)
        for i in range(n):
            if (not mask[i]):
                continue

            bigdx = self._heap_near(dashline[i].center[1], len(img)) * dashline[i].dx
            bigdy = self._heap_near(dashline[i].center[1], len(img)) * dashline[i].dy

            nearelement = self.Element(dashline[i].center, bigdx, bigdy)

            # img = printelement(img, dashline[i], (0, 0, 255))
            # img = printelement(img, nearelement, (0, 255, 255))

            cnt_in_big_radius = 0
            unique = True

            # cv2.circle(img, (int(dashline[i][0][0]), int(dashline[i][0][1])), int(Radius), (255, 255, 0))
            for j in range(n):
                if i == j:
                    continue
                # убирание куч

                if (self._pix_in_element(dashline[j].center, nearelement)):
                    cnt_in_big_radius += 1

                if cnt_in_big_radius >= MAX_CNT_IN_BIG_RADIUS:
                    mask[i] = False
                    break
                # центр внутри маленького радиуса

                if self._pix_in_element(dashline[j].center, dashline[i]):
                    mask[i] = False
                    mask[j] = False
                    pass
                # забавная оптимизация

                if self._intercect(dashline[j], nearelement):
                    unique = False
                    pass

            if unique:
                mask[i] = False
        # return mask

    def _delete_heaps(self, dashline, img):
        new = []

        n = len(dashline)
        mask = [True for i in range(n)]

        self._help_delete_heaps(dashline, img, mask)

        for i in range(n):
            if (mask[i]):
                new.append(dashline[i])

        return new

    def _dominant_color_is_black(self, img, el):
        # print(len(img))
        # print(len(img[0]))
        # print(len(img) * len(img[0]) * 3)

        curimg = img.reshape(len(img) * len(img[0]), 3)

        cntWhite = cv2.countNonZero(curimg) // 3
        maxWhite = len(img[0]) * el.dx * 2

        if (cntWhite > maxWhite):
            return False
        return True


    def _delete_dash_line_with_bright_background(self, dashline, img):
        ans = []
        for el in dashline:
            bigdx = self._heap_near(el.center[1], len(img)) * el.dx
            bigdy = self._heap_near(el.center[1], len(img)) * el.dy
            curimg = img[max(0, int(el.center[1] - bigdy)):min(len(img), int(el.center[1] + bigdy)),
                     max(0, int(el.center[0] - bigdx)):min(len(img[0]), int(el.center[0] + bigdx))]
            if self._dominant_color_is_black(curimg, el):
                ans.append(el)
        return ans

    def _detect_dashline(self, img):
        previousDashline = self.lastdashline
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold_image = cv2.threshold(gray, 150, 255, 0)

        # viewImage(threshold_image, "Чёрно-белый пёсик")
        contours, h = cv2.findContours(threshold_image, 1, 2)
        threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2RGB)
        first_image = np.copy(threshold_image)

        dashline = self._filter_contours(contours, threshold_image)
        #dashline = self._delete_dash_line_with_bright_background(dashline, threshold_image)

        dashline = self._delete_heaps(dashline, threshold_image)
        dashline = self._delete_heaps(dashline, threshold_image)
        for i in dashline:
            # cv2.circle(threshold_image, (int(i[0][0]), int(i[0][1])), int(i[1]), (51, 255, 102), thickness=2)
            pass

        for i in previousDashline:
            # threshold_image = printelement(threshold_image, i, (50, 150, 100))
            # cv2.circle(threshold_image, (int(i[0][0]), int(i[0][1])), int(i[1]), (50, 150, 100), thickness=2)
            pass

        ansDashline = []

        # for el in previousDashline:
        #     threshold_image = printelement(threshold_image, el, (255, 0, 0))
        #
        # for el in dashline:
        #     threshold_image = printelement(threshold_image, el, (0, 0, 255))
        if self.use_probability_filter:
            for a in dashline:
                isGood = False
                for b in previousDashline:
                    rad = MAXDIST_BETWEEN_ELEMENTS_IN_SEQUENT_FRAMES
                    # cv2.circle(threshold_image, (int(a[0][0]), int(a[0][1])), int(rad), (0, 255, 255), thickness=2)
                    if (self._distPixels(a.center, b.center) <= rad):
                        # cv2.circle(threshold_image, (int(a[0][0]), int(a[0][1])), int(rad), (0, 255, 255), thickness= 2)
                        isGood = True

                if not isGood and (random.randint(0, 20) != 1) and len(previousDashline) != 0:
                    continue
                    pass
                ansDashline.append(a)
        else:
            ansDashline = dashline[:]

        for i in ansDashline:
            threshold_image = self._printelement(threshold_image, i, (0, 0, 255))
            pass
        if self.use_probability_filter:
            self.lastdashline = ansDashline
        return threshold_image, ansDashline

    def _draw_size_circles(self, img):
        for i in range(5, int(len(img) * PARTOFINTEREST), 15):
            r = self._maxRadius(i, len(img))
            cv2.circle(img, (int(len(img[0]) * 3) // 4, len(img) - i), int(r), (0, 0, 255), thickness=2)
        return img

    def _make_undistorted_image(self, img):
        return cv2.undistort(img, CAMERA_MATRIX, DISTORTION_COEFFFICIENTS)

    def _sum(self, pix1, pix2):
        return [pix1[0] + pix2[0], pix1[1] + pix2[1]]

    def formated_answer(self, img):
        #img = self._make_undistorted_image(img)
        new_img, dashline = self._detect_dashline(img)
        result = []
        lenvector = 6

        for el in dashline:
            minel = self.Element([0, 0], 0, 0)
            for el2 in dashline:
                if 1 < self._distPixels(el2.center, el.center) < self._distPixels(el.center, minel.center):
                    minel = el2
            vector = [minel.center[0] - el.center[0], minel.center[1] - el.center[1]]
            len = sqrt(vector[0]**2 + vector[1]**2)
            vector = [(vector[0] / len * lenvector), (vector[1] / len * lenvector)]
            point1 = self._sum(el.center, vector) #считаем концы линии элемента разметки
            point2 = self._sum(el.center, [vector[0]*-1, vector[1]*-1])
            normal = [vector[1] / lenvector, vector[0] / lenvector] #считаем нормаль длины 1 (это вектор из начала координат)
            result.append([point1, point2, normal])


        return new_img, dashline, result

# def job(kek):
#     photos = kek[1]
#     folder = kek[0]
#     for i in photos:
#         ans = i.rfind('.')
#         ans = i[:ans] + 'edited_3' + i[ans:]
#         img = cv2.imread(folder + '\\' + i)
#         img, _ = detect_dashline(img)
#         cv2.imwrite('NewResult\\' + ans, img)
#         print('yay ' + i)


# a = input('video если флексим видео: ')
# if (a == 'video'):
#     for i in os.listdir('videos'):
#         if (i == 'test2.mp4'):
#             continue
#         video = cv2.VideoCapture('videos\\' + i)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         ans = i.rfind('.')
#         ans = i[:ans] + '_edited' + i[ans:]
#         out = cv2.VideoWriter('result_videos\\' + ans, fourcc, 30.0, (int(video.get(3)), int(video.get(4))))
#         cnt = 0
#         lastDashline = []
#         while video.isOpened():
#             ret, img = video.read()
#             if ret:
#                 # alpha = (1 - (MINPIXEL + 10) / MAXPIXEL) / (len(img) * PARTOFINTEREST)
#                 img, lastDashline = detect_dashline(img, lastDashline[:])
#                 #img = draw_size_circles(img)
#                 out.write(img)
#                 # if 0xFF == ord('q'):
#                 #     break
#             else:
#                 break
#             print('yay', cnt)
#             cnt += 1
#         out.release()
#         video.release()
# else:
#     lastDashline = []
#     for i in os.listdir('newData'):
#         ans = i.rfind('.')
#         ans = i[:ans] + 'probability_filter_ON' + i[ans:]
#         img = cv2.imread('newData\\' + i)
#         img = make_undistorted_image(img)
#         img, lastDashline = detect_dashline(img, lastDashline[:])
#         #img = draw_size_circles(img)
#         #img = drawNearRectangles(img, lastDashline)
#         cv2.imwrite('newResult\\' + ans, img)
#         #plt.imshow(img)
#         #plt.show()
#
#         print('yay ' + i)
#


kek = False
linedetector = LineDetector(use_probability_filter=False)
for i in os.listdir('newData2'):
    ans = i.rfind('.')
    ans = i[:ans] + 'withAllFilters' + i[ans:]
    img = cv2.imread('newData2\\' + i)
    if (not kek):
        kek = True
        #cv2.imwrite('newResult\\gray.jpg', cv2.cvtColor(linedetector._make_undistorted_image(img), cv2.COLOR_BGR2GRAY))
        #ret, threshold_image = cv2.threshold(cv2.cvtColor(linedetector._make_undistorted_image(img), cv2.COLOR_BGR2GRAY), 150, 255, 0)
        ret, threshold_image = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 110, 255, 0)
        #cv2.imwrite('newResult\\bw.jpg', threshold_image)
    img, lastDashline, result = linedetector.formated_answer(img)
    for a in result:
        cv2.line(img, (int(a[0][0]), int(a[0][1])), (int(a[1][0]), int(a[1][1])), (0, 255, 0), thickness= 2)

    #img = draw_size_circles(img)
    #img = drawNearRectangles(img, lastDashline)
    cv2.imwrite('newResult2\\' + ans, img)
    #plt.imshow(img)
    #plt.show()
    print(i, ans)
# viewImage(img, 'kek')
# plt.imshow(img)
