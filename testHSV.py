import cv2
import numpy as np

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 滑鼠左鍵點擊
        pixel = hsv[y, x]
        print(f"選擇的 HSV 值: {pixel}")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 轉換為 HSV
    cv2.imshow("HSV", hsv)

    cv2.setMouseCallback("HSV", pick_color)  # 滑鼠點擊偵測顏色

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
        break

cap.release()
cv2.destroyAllWindows()
