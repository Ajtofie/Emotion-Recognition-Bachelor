import math
import cv2
import numpy as np
from src.EmotionRecognition.processor import Detector
from colorama import Fore


def process_image(image):
    image = pre_process(image)
    faces = Detector.detect_face_from(image)
    '''Get Landmarks'''
    for face in faces:
        landmarks = Detector.get_all_facial_landmarks_from(image, face)
        if not landmarks:
            print(Fore.RED + "No landmarks found")
            return None, False

        '''If eyes are NOT horizontal rotate image so that they are'''
        left_eye, right_eye = Detector.get_eye_position(landmarks)
        if not eyes_are_horizontal(left_eye, right_eye):
            image = rotate_image_by_angle(image, get_angle(left_eye, right_eye))

        '''If nose is not in the center, move the image'''
        nose_position = Detector.get_specific_facial_landmarks(landmarks, [34])[0]
        if not nose_centered(image, nose_position):
            image = move_image(image, nose_position)

    return image, True


# resizing, convert to grayscale, applying clahe
def pre_process(image):
    image_resized = resizing_image(image)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)


def resizing_image(image):
    return cv2.resize(image, (256, 256))


def eyes_are_horizontal(left_eye, right_eye):
    return left_eye[1] is right_eye[1]


def rotate_image_by_angle(image, angle):
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    rm = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.25)

    return cv2.warpAffine(src=image, M=rm, dsize=(height, width))


def get_angle(left_eye, right_eye):
    horizontal_gradient = (1, 0)
    eye_gradient = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    angle = calculate_angle(horizontal_gradient, eye_gradient)

    if left_eye[1] > right_eye[1]:
        angle = -angle

    return angle


def calculate_angle(vector1, vector2):
    abs_eyes = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    angle = math.acos(vector2[0] / abs_eyes)
    return math.degrees(angle)


def nose_centered(image, point):
    height, width = image.shape[:2]
    desired_nose_position = (width / 2, height / 2)
    if point is desired_nose_position:
        return True
    return False


def move_image(image, nose_position):
    (height, width) = image.shape[:2]
    desired_nose_position = (width / 2, height / 2)
    diff_x = get_difference_between_points(nose_position[0], desired_nose_position[0])
    diff_y = get_difference_between_points(nose_position[1], desired_nose_position[1])
    M = np.float32([[1, 0, diff_x],
                    [0, 1, diff_y]])
    return cv2.warpAffine(image, M, (width, height))


def get_difference_between_points(point1, point2):
    difference = abs(point1 - point2)
    if point1 > point2:
        return -difference
    else:
        return difference


def crop_image(image):
    height, width = image.shape
    print image.shape
    start_row, start_col = int(height * .2), int(width * .2)
    end_row, end_col = int(height * .75), int(width * .75)
    cropped = image[start_row:end_row, start_col:end_col]
    return cropped
