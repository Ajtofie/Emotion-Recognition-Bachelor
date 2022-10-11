# coding=utf-8
import os
from src.EmotionRecognition.processor import EmotionLoader
import numpy as np
from skimage import io
from colorama import Fore, Style
from src.EmotionRecognition.processor import ImageProcessor


def load_images_from_data_path(data_path):
    pre_processed_images = []
    fully_processed_images = []
    emotion_list = []
    for img in os.listdir(data_path):
        print(Style.RESET_ALL)
        if "jpg" not in img:
            print (Fore.RED + "Not an image: " + img)
            continue

        '''Loading Images'''
        print("Loading image: " + img)
        image = io.imread(data_path + "/" + img)

        '''Process Image'''
        print("Processing image ...")
        processed_img, process_successful = ImageProcessor.process_image(image)
        if not process_successful:
            print(Fore.RED + "Image couldn't be processed!")
            continue
        pre_processed_images.append(ImageProcessor.pre_process(image))
        fully_processed_images.append(processed_img)

        '''Loading Emotions'''
        print("Loading emotion ...")
        emotion_list.append(EmotionLoader.get_emotion(img))


        # todo wenn image nicht processed wird, wird das ganze bild zurückgegeben sollte net so sein

        print("Finished loading image: " + "\n------------------------------------------------")

    return np.array(pre_processed_images), np.array(fully_processed_images), np.array(emotion_list)
