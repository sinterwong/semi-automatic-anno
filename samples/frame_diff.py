import cv2

# 读取视频文件或摄像头
video_capture = cv2.VideoCapture(0)

# 读取第一帧作为背景帧
_, background = video_capture.read()
background = cv2.blur(background, (3, 3), (-1, -1))
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

while True:
    # 读取当前帧
    ret, frame = video_capture.read()
    if not ret:
        break
      
    frame = cv2.blur(frame, (3, 3), (-1, -1))

    # 将当前帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算当前帧与背景帧的差分图像
    frame_diff = cv2.absdiff(background_gray, gray_frame)

    # 对差分图像进行阈值化处理
    _, thresholded_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # 显示阈值化后的差分图像
    cv2.imshow("Frame Difference", thresholded_diff)

    background_gray = gray_frame

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流和窗口
video_capture.release()
cv2.destroyAllWindows()
