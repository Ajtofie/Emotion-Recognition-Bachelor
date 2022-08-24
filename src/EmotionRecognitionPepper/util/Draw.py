

def draw_landmarks():
    for a in input_img_landmarks:
        for key, values in a.items():
            for (x, y) in values:
                cv2.circle(input_img_grey, (x, y), 1, (0, 0, 255), 1)


def show_image(name, img):
    draw_landmarks()
    cv2.imshow(name, img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()                