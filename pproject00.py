#!/usr/bin/python

# **********************************************
# * Hand Gesture Recognition Implementation v1.0
# * 2 July 2016
# * Mahaveer Verma
# **********************************************
import requests,json
import urllib.parse
from time import strftime,gmtime

import cv2
import numpy as np
import math
#from GestureAPI import *
from GestureAPI1 import *

# Variables & parameters
hsv_thresh_lower = 150
gaussian_ksize = 11
gaussian_sigma = 0
morph_elem_size = 13
median_ksize = 3
capture_box_count = 9
capture_box_dim = 20
capture_box_sep_x = 8
capture_box_sep_y = 18
capture_pos_x = 500
capture_pos_y = 150
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
finger_thresh_l = 2.0
finger_thresh_u = 3.8
radius_thresh = 0.04  # factor of width of full frame
first_iteration = True
finger_ct_history = [0, 0]

i=0
# ------------------------ Function declarations ------------------------ #

'''
def convex(frame_in,hand_hist):
    frame_in = cv2.medianBlur(frame_in, 3)
    gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)

    gray[0:int(cap_region_y_end * gray.shape[0]), 0:int(0.75 * gray.shape[1])] = 0  # Right half screen only
    gray[int(cap_region_y_end * gray.shape[0]):gray.shape[0], 0:gray.shape[1]] = 0
    # back_projection = cv2.calcBackProject([hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)


    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_elem_size, morph_elem_size))
    # cv2.filter2D(back_projection, -1, disc, back_projection)
    # back_projection = cv2.GaussianBlur(back_projection, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    # back_projection = cv2.medianBlur(back_projection, median_ksize)
    ret, thresh = cv2.threshold(gray,128, 255, 0)
    cv2.imshow('', thresh)
    cv2.waitKey(0)
    ret,contours,hireachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt,returnPoints=False)
    defects = cv2.convexityDefects(hull)
    dist = 0
    x=0
    y=0
    cv2.imshow('',frame_in)
    cv2.waitKey(0)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.circle(frame_in, far, 5, [0, 0, 255], -1)
        temp = dist(start,end)
        if(temp>dist):
            dist = temp
            x = start
            y = end
    return x,y
    
'''
# 1. Hand capture histogram
def hand_capture(frame_in, box_x, box_y): # (frame_original , ... )
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)



    ROI = np.zeros([capture_box_dim * capture_box_count, capture_box_dim, 3], dtype=hsv.dtype) # (20*9 , 20 ,3 )
    for i in range(capture_box_count):
        ROI[i * capture_box_dim:i * capture_box_dim + capture_box_dim, 0:capture_box_dim] = hsv[box_y[i]:box_y[ # box_x/y just tell location
                                                                                                             i] + capture_box_dim,
                                                                                            box_x[i]:box_x[
                                                                                                         i] + capture_box_dim]
    hand_hist = cv2.calcHist([ROI], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\decisionTree\\ROI.jpg',ROI)
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('C:\\Users\\bkpor\\Dropbox\\_project\\proj1\\output\\ROI9chan.jpg',ROI)
    return hand_hist


# 2. Filters and threshold
def hand_threshold(frame_in, hand_hist): # entire fg_frame cut out bg
    frame_in = cv2.medianBlur(frame_in, 3)

    time = strftime('%H_%M_%S_%d%b%Y', gmtime())

    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\bfCutHSV' + time + '.jpg',
                hsv)

    cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\toHandThes' + time + 'hsv.jpg',
                frame_in)

    hsv[0:int(cap_region_y_end * hsv.shape[0]), 0:int(cap_region_x_begin * hsv.shape[1])] = 0  # Right half screen only
    hsv[int(cap_region_y_end * hsv.shape[0]):hsv.shape[0], 0:hsv.shape[1]] = 0

    cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\hsv'+time+'.jpg', hsv)
    # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\decisionTree\\60cm\\60cm_'+time+'hsv.jpg', hsv)

    back_projection = cv2.calcBackProject([hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)

    cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\backproject' + time + '.jpg',
                back_projection)

    #white object(hand)

    #cv2.waitKey(0)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_elem_size, morph_elem_size))
    cv2.filter2D(back_projection, -1, disc, back_projection)

    # time = strftime('%H_%M_%S_%d%b%Y', gmtime())
    # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\dialation'+time+'.jpg', back_projection)

    back_projection = cv2.GaussianBlur(back_projection, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    back_projection1 = cv2.medianBlur(back_projection,median_ksize)
    ret, thresh = cv2.threshold(back_projection  ,50, 255, 0)
    cv2.imshow('',thresh)

    # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\decisionTree\\60cm\\60cm_backproeject'+time+'.jpg', back_projection)
    # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\thresh'+time+'.jpg', thresh)

    return thresh


# 3. Find hand contour
def hand_contour_find(contours):
    max_area = 0
    largest_contour = -1

    print(contours)

    for i in range(len(contours)):
        cont = contours[i]

        print(cont)

        area = cv2.contourArea(cont)
        if (area > max_area):
            max_area = area
            largest_contour = i
    if (largest_contour == -1):
        return False, 0
    else:
        h_contour = contours[largest_contour]
        return True, h_contour


# 4. Detect & mark fingers
def mark_fingers(frame_in, hull, pt, radius):
    global first_iteration
    global finger_ct_history
    finger = [(hull[0][0][0], hull[0][0][1])]
    j = 4

    cx = pt[0]
    cy = pt[1]

    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i + 1][0][0]) ** 2 + (hull[-i][0][1] - hull[-i + 1][0][1]) ** 2)
        if (dist > 18):
            if (j == 0):
                finger = [(hull[-i][0][0], hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0], hull[-i][0][1]))
            j = j + 1

    temp_len = len(finger)
    i = 0
    while (i < temp_len):
        dist = np.sqrt((finger[i][0] - cx) ** 2 + (finger[i][1] - cy) ** 2)
        if (dist < finger_thresh_l * radius or dist > finger_thresh_u * radius or finger[i][1] > cy + radius):
            finger.remove((finger[i][0], finger[i][1]))
            temp_len = temp_len - 1
        else:
            i = i + 1

    temp_len = len(finger)
    if (temp_len > 5):
        for i in range(1, temp_len + 1 - 5):
            finger.remove((finger[temp_len - i][0], finger[temp_len - i][1]))

    palm = [(cx, cy), radius]

    if (first_iteration):
        finger_ct_history[0] = finger_ct_history[1] = len(finger)
        first_iteration = False
    else:
        finger_ct_history[0] = 0.34 * (finger_ct_history[0] + finger_ct_history[1] + len(finger))

    if ((finger_ct_history[0] - int(finger_ct_history[0])) > 0.8):
        finger_count = int(finger_ct_history[0]) + 1
    else:
        finger_count = int(finger_ct_history[0])

    finger_ct_history[1] = len(finger)

    count_text = "FINGERS:" + str(finger_count)
    cv2.putText(frame_in, count_text, (int(0.62 * frame_in.shape[1]), int(0.88 * frame_in.shape[0])),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

    for k in range(len(finger)):
        # cv2.drawContours(frame_in,[finger[k]],-1,(255,0,0),2)
        cv2.circle(frame_in, finger[k], 10, (255,255,255), 2)
        line_center_to_finger = cv2.line(frame_in, finger[k], (cx, cy), 255, 2)

    return frame_in, finger, palm


# 5. Mark hand center circle

def mark_hand_center(frame_in, cont):
    max_d = 0
    pt = (0, 0)

    x, y, w, h = cv2.boundingRect(cont)
    cv2.rectangle(frame_in, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.drawContours(frame_in,[rect],0,(0,255,0),2)


    for ind_y in range(int(y + 0.25 * h),#0.3 --> 0.8
                        int(y + 0.6 * h)):  # around 0.25 to 0.6 region of height (Faster calculation with ok results)
        for ind_x in range(int(x + 0.3 * w),#0.3-->0.6
                            int(x + 0.6* w)):  # around 0.3 to 0.6 region of width (Faster calculation with ok results)
            dist = cv2.pointPolygonTest(cont, (ind_x, ind_y), True)
            if (dist > max_d):
                max_d = dist
                pt = (ind_x, ind_y)
    if (max_d > radius_thresh * frame_in.shape[1]):
        thresh_score = True
        cv2.circle(frame_in, pt, int(max_d), (255, 255, 0), 2)
    else:
        thresh_score = False
    return frame_in, pt, max_d, thresh_score


# 6. Find and display gesture
#
def find_gesture(frame_in, finger, palm):
    frame_gesture.set_palm(palm[0], palm[1])
    frame_gesture.set_finger_pos(finger)
    frame_gesture.calc_angles() # charac 1
    frame_gesture.fingerLength()
    frame_gesture.betwLength()#charac 2
    gesture_found = DefineAndDecide(frame_gesture)
    gesture_text = "GESTURE:" + ''.join(gesture_found)


    time = strftime('%H_%M_%S_%d%b%Y', gmtime())
    if gesture_found is not 'NONE':
        cv2.putText(frame_in, gesture_text, (int(0.56 * frame_in.shape[1]), int(0.97 * frame_in.shape[0])),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)
        cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\_finished' + time + '.jpg',frame_in)
        cv2.putText(frame_in, 'Correct!!', (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3, 9)
        # if gesture_found in ['1VGes','2LGes','3CabowGes','4IndexRing']:



    
    #global time
    #time = strftime('%H_%M_%S_%d%b%Y', gmtime())


    return frame_in, gesture_found


# 7. Remove bg from image

def remove_bg(frame):
    time = strftime('%H_%M_%S_%d%b%Y', gmtime())

    fg_mask = bg_model.apply(frame)

    # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\New folder\\frameNOand' + time + '.jpg', frame)

    cv2.imshow('',fg_mask)

    kernel = np.ones((3, 3), np.uint8)
    fg_mask1 = cv2.erode(fg_mask, kernel, iterations=1)
    frame = cv2.bitwise_and(frame, frame, mask=fg_mask1)

    cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\bgsubtract' + time + '.jpg', fg_mask)
    cv2.imwrite('C:\\Users\\bkpor\\PycharmProqjects\\imgProcess_ject2\\PORPREPRO\\erode' + time + '.jpg', fg_mask1)
    cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\frame_and' + time + '.jpg',frame)
    # cv2.imwrite('C:\\Users\\bkpor\\Dropbox\\_project\\proj1\\output\\herode.jpg', fg_mask1)
    return frame




# ------------------------ BEGIN ------------------------ #

capture_done = 0
bg_captured = 0
# Camera
camera = cv2.VideoCapture(0)


while (1):
    # Capture frame from camera
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Operations on the frame
    frame = cv2.flip(frame, 1)

    # frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 255), 1)
    frame_original = np.copy(frame)
    if (bg_captured):
        fg_frame = remove_bg(frame)

    if (not (capture_done and bg_captured)):
        if (not bg_captured):
            cv2.putText(frame, "Remove hand from the frame and press 'b' to capture background",
                        (int(0.05 * frame.shape[1]), int(0.97 * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, 8)
            # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\decisionTree\\start.jpg',frame)
        else:
            cv2.putText(frame, "Place hand inside boxes and press 'c' to capture hand histogram",
                        (int(0.08 * frame.shape[1]), int(0.97 * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, 8)

        first_iteration = True
        finger_ct_history = [0, 0]
        box_pos_x = np.array([capture_pos_x, capture_pos_x + capture_box_dim + capture_box_sep_x,
                              capture_pos_x + 2 * capture_box_dim + 2 * capture_box_sep_x, capture_pos_x,
                              capture_pos_x + capture_box_dim + capture_box_sep_x,
                              capture_pos_x + 2 * capture_box_dim + 2 * capture_box_sep_x, capture_pos_x,
                              capture_pos_x + capture_box_dim + capture_box_sep_x,
                              capture_pos_x + 2 * capture_box_dim + 2 * capture_box_sep_x], dtype=int)
        box_pos_y = np.array(
            [capture_pos_y, capture_pos_y, capture_pos_y, capture_pos_y + capture_box_dim + capture_box_sep_y,
             capture_pos_y + capture_box_dim + capture_box_sep_y, capture_pos_y + capture_box_dim + capture_box_sep_y,
             capture_pos_y + 2 * capture_box_dim + 2 * capture_box_sep_y,
             capture_pos_y + 2 * capture_box_dim + 2 * capture_box_sep_y,
             capture_pos_y + 2 * capture_box_dim + 2 * capture_box_sep_y], dtype=int)
        for i in range(capture_box_count):
            cv2.rectangle(frame, (box_pos_x[i], box_pos_y[i]),
                          (box_pos_x[i] + capture_box_dim, box_pos_y[i] + capture_box_dim), (255,0,0), 1)
        # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\decisionTree\\cap.jpg', frame)
    else:

        frame = hand_threshold(fg_frame, hand_histogram)

        #cut bg only white hand
        # cv2.imshow('',frame)
        # cv2.waitKey(0)

        contour_frame = np.copy(frame)
        # cut bg left black hand(threshold)
        # cv2.imshow('',contour_aframe)
        # cv2.waitKey(0)

        ret,contours, hierarchy = cv2.findContours(contour_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # cv2.imshow('',frame)
        # cv2.waitKey(0)



        #find the largest contours , found = TRUE
        found, hand_contour = hand_contour_find(contours)

        # frame1 =np.ones(frame.shape,np.uint8)
        # cv2.drawContours(frame1,[hand_contour],-1,(255,255,255),2)
        # cv2.imshow('',frame1)

        #time = strftime('%H_%M_%S_%d%b%Y', gmtime())
        # cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\contours' + time + '.jpg',
        #             frame1)


        if (found): # hand_convex_hull is ลิสต์ของตำแหน่งอาเรย์สองมิติ ที่เป็นจุด convex(จุดปลายนิ้วมือ)
            hand_convex_hull = cv2.convexHull(hand_contour)
            #pointOne, pointTwo = convex(frame_original, hand_histogram)
            frame, hand_center, hand_radius, hand_size_score =mark_hand_center(frame_original, hand_contour)

            #entire pic frame original
            # cv2.imshow('',frame_original)

            # cv2.waitKey(0)
            frame, finger, palm = mark_fingers(frame, hand_convex_hull, hand_center, hand_radius)

            if (len(finger) == 2):
                frame_gesture = DecisionTreeGesture("frame_gesture")

                # k = str(len(finger))
                #
                # f = open('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\new.txt', 'a')
                # f.write('\n\n')
                # f.write(k + ' finger :')
                # for i in range(
                #         len(finger)):
                #     fingerStr = '(' + str(finger[i][0]) + ',' + str(finger[i][1]) + ')'
                #     f.write(fingerStr)
                # f.write('\n')
                # centerStr = 'Center = (' + str(palm[0][0]) + ',' + str(palm[0][1]) + ')'
                # f.write(centerStr+ '\n')
                # f.write('radius = ' + str(palm[1]) + '\n')
                # time = strftime('%H_%M_%S_%d%b%Y', gmtime())
                # f.write(time)
                # f.close()
                #
                #
                time = strftime('%H_%M_%S_%d%b%Y', gmtime())
                k = str(len(finger))

                f = open('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\PORPREPRO\\record.txt', 'a')
                f.write('\n\n')

                # palm[0][0] = cx
                # palm[0][1] = cy
                cx = palm[0][0]
                cy = palm[0][1]

                angle = np.zeros(len(finger), dtype=int)
                for i in range(len(finger)):
                    y = finger[i][1]
                    x = finger[i][0]
                    angle[i] = abs(math.atan2((cy - y), (x - cx))* 180 / math.pi)
                ang = angle[1] - angle[0]
                # x= fing1[0][0]  y=fing1[0][1]  x=fing2[1][0]  y=fing2[1][1]
                betwLen = np.sqrt(np.square(finger[0][0] - finger[1][0]) + np.square(finger[0][1] - finger[1][1]))
                fing1 = np.sqrt(np.square(finger[0][0] - cx) + np.square(finger[0][1] - cy))
                fing2 = np.sqrt(np.square(finger[1][0] - cx) + np.square(finger[1][1] - cy))

                f.write(str(ang))
                f.write(',')
                f.write(str(fing1 / betwLen))
                f.write(',')
                f.write(str(fing2 / betwLen))
                f.write('\n')


                f.write(time)
                f.close()

                frame, gesture_found = find_gesture(frame, finger, palm)

        else:
            frame = frame_original


    # ret, frame= cv2.threshold(frame,50,255,0)
    # Display frame in a window
               # cv2.circle(frame, (475, 225), 45, (0, 0, 255), 1)
            # cv2.circle(frame,(490,90),3,(0,0,255),1)
            # cv2.circle(frame,(415,105),3,(0,0,255),1)
            #
            # cv2.circle(frame,(475, 225),50,(255,255,0),1)
            # cv2.circle(frame ,(450, 62),3,(255,255,0),1)
            # cv2.circle(frame, (345, 200), 3, (255, 255, 0), 1)

    # cv2.circle(frame, (345, 200), 1, (255, 255, 255), 1)
    # cv2.circle(frame, (345, 200), 1, (255, 255, 255), 1)

    cv2.imshow('Hand Gesture Recognition v1.0', frame)

    interrupt = cv2.waitKey(10)

    # Quit by pressing 'q'
    if interrupt & 0xFF == ord('q'):
        # cv2.imwrite('C:\\Users\\bkpor\\Dropbox\\_project\\proj1\\output\\frame.jpg', frame)
        break
    # Capture hand by pressing 'c'
    elif interrupt & 0xFF == ord('c'):
        if (bg_captured):
            capture_done = 1
            hand_histogram = hand_capture(frame_original, box_pos_x, box_pos_y)
    # Capture background by pressing 'b'
    elif interrupt & 0xFF == ord('b'):
        bg_model = cv2.createBackgroundSubtractorMOG2()

        bg_captured = 1
    # Reset captured hand by pressing 'r'
    elif interrupt & 0xFF == ord('r'):
        capture_done = 0
        bg_captured = 0

    elif interrupt & 0xFF == ord('W'):
        cv2.imwrite('C:\\Users\\bkpor\\PycharmProjects\\imgProcess_ject2\\decisionTree\\first.jpg', frame)
        break

camera.release()
cv2.destroyAllWindows()
#

# LINE_ACCESS_TOKEN="QprbgIQjZ3o3wltuKpDbtqArbytRy8O3bLpFR24loZq"
# url = "https://notify-api.line.me/api/notify"
#
#
# message ="User is dangerous!" # ข้อความที่ต้องการส่ง
# msg = urllib.parse.urlencode({"message":message})
# LINE_HEADERS = {'Content-Type':'application/x-www-form-urlencoded',"Authorization":"Bearer "+LINE_ACCESS_TOKEN}
# session = requests.Session()
# a=session.post(url, headers=LINE_HEADERS, data=msg)
# print(a.text)