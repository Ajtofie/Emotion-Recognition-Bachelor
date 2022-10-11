import dlib

path = "C:\Users\Matej\IdeaProjects\Bachelorarbeit\\resources\shape_predictor\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)


def detect_face_from(image):
    return detector(image)


def get_all_facial_landmarks_from(image, face):
    return predictor(image, face)


def get_specific_facial_landmarks(landmarks, points):
    # returns specific landmarks features with the index in points
    landmark_list = []
    for n in points:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_list.append((x, y))
    return landmark_list


def get_all_landmark_coordinates(landmarks):
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
    return points


def get_eye_position(landmarks):
    eyes = get_specific_facial_landmarks(landmarks, [36, 45])
    return eyes[0], eyes[1]

