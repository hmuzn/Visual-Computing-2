import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

def create_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def create_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], cv2.pyrUp(gaussian_pyramid[i], dstsize=size))
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid

hand_img = cv2.imread('hand.jpg')
eye_img = cv2.imread('eye.jpg')

levels = 3

hand_gaussian_pyramid = create_gaussian_pyramid(hand_img, levels)
eye_gaussian_pyramid = create_gaussian_pyramid(eye_img, levels)

hand_laplacian_pyramid = create_laplacian_pyramid(hand_gaussian_pyramid)
eye_laplacian_pyramid = create_laplacian_pyramid(eye_gaussian_pyramid)

eye_mask = np.zeros_like(eye_img)
eye_mask = cv2.circle(eye_mask, (eye_mask.shape[1] // 2, eye_mask.shape[0] // 2), eye_mask.shape[0] // 2, (255, 255, 255), -1)

mask_gaussian_pyramid = create_gaussian_pyramid(eye_mask, levels)

blended_pyramid = []
for i in range(levels):
    eye_lap = cv2.resize(eye_laplacian_pyramid[i], (hand_laplacian_pyramid[i].shape[1], hand_laplacian_pyramid[i].shape[0]))
    mask = cv2.resize(mask_gaussian_pyramid[i], (hand_laplacian_pyramid[i].shape[1], hand_laplacian_pyramid[i].shape[0]), interpolation=cv2.INTER_NEAREST)
    blended = cv2.add(hand_laplacian_pyramid[i] * (1 - mask), eye_lap * mask)
    blended_pyramid.append(blended)

blended_img = blended_pyramid[0]
for i in range(1, levels):
    blended_img = cv2.pyrUp(blended_img)
    blended_img = cv2.add(blended_img, blended_pyramid[i])

# 결과를 저장
cv2.imwrite("결과1.jpg", blended_img)

# 최종 결과 출력.
plt.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
plt.title("Blended Image")
plt.axis('on') 
plt.grid(False)
plt.show()



