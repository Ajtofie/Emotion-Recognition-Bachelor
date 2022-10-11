def get_emotion(filename):
    return get_label_from_filename(filename)


def get_label_from_filename(filename):
    index = filename.find('.')
    return filename[index + 1:index + 3]
