import numpy as np
import cv2
import sys
import os

import math

inteval = 1
frame_no = 0

if __name__ == '__main__':
    if(len(sys.argv)==1):
        cap = cv2.VideoCapture(0)
    elif(len(sys.argv)==2):
        if(sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
            cap = cv2.VideoCapture(int(sys.argv[1]))
        else:
            cap = cv2.VideoCapture(sys.argv[1])
    else:
        assert(0), "too many arguments"

    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print length

    if(os.path.exists(str(sys.argv[1][0:-4])) == False):
        os.mkdir(str(sys.argv[1][0:-4]))

    while(cap.isOpened()):
        ret, frame = cap.read()
 
        if not ret:
            break

        # cv2.imshow('vid', frame)

        frame_no = frame_no+1
        filename = os.getcwd() + '/' + str(sys.argv[1][0:-4]) + '/' + str(frame_no)+'.jpg'

        cv2.imwrite(filename,frame)

        c = cv2.waitKey(inteval) & 0xFF
        if c==27 or c==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
