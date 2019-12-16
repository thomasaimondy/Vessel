'''
摄像机角度偏移告警
'''
import cv2
import do_match
import numpy as np
from PIL import Image, ImageDraw, ImageFont

'''
告警信息
'''
def putText(frame, text):
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype("fonts/msyh.ttc", 30, encoding="utf-8")
    draw.text((50, 50), text, (0, 255, 255), font=font)

    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    return cv2_text_im




texts = ["完全偏移","严重偏移", "轻微偏移", "无偏移"]

cap = cv2.VideoCapture('videos/test4_new.mp4')

if (cap.isOpened()== False):
    print("Error opening video stream or file")

first_frame = True
pre_frame = 0

index = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        if first_frame:
            pre_frame = frame
            first_frame = False
            continue

        index += 1
        if index % 24 == 0:
            result = do_match.match2frames(pre_frame, frame)
            print("检测结果===>", texts[result])

            if result > 1:  # 缓存最近无偏移的帧
                pre_frame = frame

            size = frame.shape

            if size[1] > 720: # 缩小显示
                frame = cv2.resize(frame, (int(size[1]*0.5), int(size[0]*0.5)), cv2.INTER_LINEAR)

            text_frame = putText(frame, texts[result])

            cv2.imshow('Frame', text_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()