import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cap = cv2.VideoCapture(1)
object_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換成 HSV 色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 紅色範圍(HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 找到輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            #長方形座標(x,y為左上角座標，w表寬度，h表高度)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"({x},{y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #計算深度
            K = 1000
            z = K / (w + h) # w+h越小代表物體越遠，反之則越近
            object_positions.append((x, y, z))
   

    cv2.imshow("Tracking Red Object", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# 繪製 3D 軌跡
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 取出 x, y, z 座標
x_vals = [pos[0] for pos in object_positions]
y_vals = [pos[1] for pos in object_positions]
z_vals = [pos[2] for pos in object_positions]

ax.plot(x_vals, y_vals, z_vals, label="Object Path")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis (Depth)")
ax.legend()
plt.show()