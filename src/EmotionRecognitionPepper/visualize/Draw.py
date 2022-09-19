import time

import cv2

# def draw_landmarks():
#     for a in input_img_landmarks:
#         for key, values in a.items():
#             for (x, y) in values:
#                 cv2.circle(input_img_grey, (x, y), 1, (0, 0, 255), 1)


def show_image(name, img):
    # draw_landmarks()
    cv2.imshow(name, img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def draw_landmarks_to(image, landmarks):
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)


def draw_eye_landmarks_to(image, landmarks):
    for n in [36, 45]:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 3, (0 ,0 ,255), -1)


def draw_specific_landmars_to(image, landmarks, points):
    for n in points:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)