import cv2
import numpy as np
import math

def ROI(image):
    roi = np.array([
        [(200,400) , (700,400) , (1280,660) , (0,660)]
    ])
    mask = np.zeros_like(image)
    image2 = cv2.fillPoly(mask, roi ,255)
    masked_image = cv2.bitwise_and(image,image2)
    return masked_image

def show_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for lines2 in lines:
            x1 , y1 , x2 , y2 = lines2.reshape(4)
            if (y2 - y1)/(x2-x1) > math.tan(math.pi/12) or (y2 - y1)/(x2-x1) < -math.tan(math.pi/12):
                cv2.line(line_image, (x1,y1) , (x2,y2), (255,0,0), 10)
    return line_image


cap = cv2.VideoCapture('Automathon-challenge-CV.mp4')

while(cap.isOpened):
        # 800x600 windowed mode
        _ , imgfinal = cap.read()
        img = np.copy(imgfinal)
        img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)   #Gray
        #img = cv2.GaussianBlur(img, (5, 5), 0)         #GaussianBlur
        img = cv2.Canny(img, 200, 150)              # Canny function contains Gaussian blur
        #ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = ROI(img)
        lines = cv2.HoughLinesP(img, 2, np.pi / 180, 50, np.array([]), minLineLength=20, maxLineGap = 5)
        lane_img = show_lines(imgfinal , lines)
        combo_img = cv2.addWeighted(imgfinal , 0.8, lane_img, 1, 1)

        cv2.imshow('img' , combo_img)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280 , 720))
        out.write(combo_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


