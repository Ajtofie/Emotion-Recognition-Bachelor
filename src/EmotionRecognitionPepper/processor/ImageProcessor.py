import math
import cv2
import numpy as np
from src.EmotionRecognitionPepper.visualize import Draw
from src.EmotionRecognitionPepper.processor import Detector


def process_image(image):
    image = pre_process(image)
    rot_image = image.copy()
    faces = Detector.detect_face_from(image)
    '''Get Landmarks'''
    for face in faces:
        landmarks = Detector.get_all_facial_landmarks_from(image, face)
        if not landmarks:
            print("No landmarks found")
            return None, False

        '''If eyes are NOT horizontal rotate image so that they are'''
        left_eye, right_eye = Detector.get_eye_position(landmarks)
        if not eyes_are_horizontal(left_eye, right_eye):
            image = rotate_image_by_angle(image, get_angle(left_eye, right_eye))  # todo get right angle

        '''If nose is not in the center, move the image'''
        nose_position = Detector.get_specific_facial_landmarks(landmarks, [34])[0]
        if not nose_centered(image, nose_position):
            image = move_image(image, nose_position)  # todo move image properly
        rot_image = image.copy()

        # Draw.show_image("rotated, moved and scaled image", image)
    return rot_image, True


# resizing, convert to grayscale, applying clahe
def pre_process(image):
    image_resized = cv2.resize(image, (256, 256))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)


def eyes_are_horizontal(left_eye, right_eye):
    return left_eye[1] is right_eye[1]


def rotate_image_by_angle(image, angle):
    scaling_factor = 1.5
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    rm = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

    return cv2.warpAffine(src=image, M=rm, dsize=(height, width))


def get_angle(left_eye, right_eye):
    horizontal_gradient = (1, 0)
    eye_gradient = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])

    angle = calculate_angle(horizontal_gradient, eye_gradient)

    if left_eye[1] > right_eye[1]:
        angle = angle * -1

    return angle


def calculate_angle(vector1, vector2):
    """Using arccos to calculate angle between two vectors"""
    scalar_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    abs_horizontal = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    abs_eyes = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    angle = math.acos(scalar_product / (abs_horizontal * abs_eyes))
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


def get_difference_between_points(point1, point2): # todo refactor + rename
    difference = abs(point1 - point2)
    if point1 > point2:
        return difference * -1
    else:
        return difference


def crop_image(image):
    height, width = image.shape
    print image.shape
    start_row, start_col = int(height * .2), int(width * .2)
    end_row, end_col = int(height * .75), int(width * .75)
    cropped = image[start_row:end_row, start_col:end_col]
    return cropped
