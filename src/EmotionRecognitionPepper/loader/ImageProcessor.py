

def get_facial_landmark_coordinates(images):
    x_list = []
    y_list = []
    for img in images:
        img_landmarks = face_recognition.face_landmarks(img)
        for a in img_landmarks:
            for key, values in a.items():
                for (x, y) in values:
                    x_list.append(float(x))
                    y_list.append(float(y))
    x_mean = np.mean(x_list)  # find both coordinates of center of gravity
    y_mean = np.mean(x_list)
    x_central = [(x-x_mean) for x in x_list]  # Calculate the distance center <-> other points in both axes
    y_central = [(y-y_mean) for y in y_list]

    landmarks_vectorized = []
    for x, y, w, z in zip(x_central, y_central, x_list, y_list): # Store all landmarks in one list in the form of x1, y1, x2, y2 etc.
        landmarks_vectorized.append(w)
        landmarks_vectorized.append(z)
        mean_np = np.asarray((y_mean, x_mean))
        coor_np = np.asarray((z, w))
        dist = np.linalg.norm(coor_np-mean_np)
        landmarks_vectorized.append(dist)
        landmarks_vectorized.append((math.atan2(y, x) * 360) / (2 * math.pi))

    data = {'landmarks_vectorized': landmarks_vectorized}

    return data