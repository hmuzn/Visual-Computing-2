import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


hand_img = cv.imread('hand.jpg')
eye_img = cv.imread('eye.jpg')

hand_center_x, hand_center_y = hand_img.shape[1] // 2, hand_img.shape[0] // 2

eye_scale = 0.25  
hand_height, hand_width = hand_img.shape[:2]

eye_aspect_ratio = eye_img.shape[1] / eye_img.shape[0]
eye_new_width = int(hand_width * eye_scale)
eye_new_height = int(eye_new_width / eye_aspect_ratio)

eye_resized = cv.resize(eye_img, (eye_new_width, eye_new_height))

eye_mask_resized = np.zeros_like(eye_resized, eye_resized.dtype)
cv.circle(eye_mask_resized, (eye_new_width // 2, eye_new_height // 2), min(eye_new_width // 2, eye_new_height // 2), (255, 255, 255), -1)

top_left_x = hand_center_x - (eye_new_width // 2)
top_left_y = hand_center_y - (eye_new_height // 2)

seamless_center = (hand_center_x, hand_center_y)

blended_img = cv.seamlessClone(eye_resized, hand_img, eye_mask_resized, seamless_center, cv.NORMAL_CLONE)

#최종 이미지 저장
cv.imwrite("결과.jpg", blended_img)

#결과 화면에 보여주기
plt.imshow(cv.cvtColor(blended_img, cv.COLOR_BGR2RGB))
plt.title("result Image")
plt.axis('on') 
plt.grid(False)
plt.show()
