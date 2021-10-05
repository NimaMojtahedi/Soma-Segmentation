# This file provides utility functions

# Necessary packages
from read_roi import read_roi_zip
import h5py
import os
import numpy as np
import cv2
from tqdm import tqdm
from detectron2.structures import BoxMode


# convert ImageJ labels to coco format dataset
class ImageJ2COCO:

    # initialize the class
    def __init__(self, image_path, label_path, output_path, start_index=0, end_index="all", image_nr=1, id_starter=10):

        # path info
        self.image_path = image_path
        self.label_path = label_path
        self.output_path = output_path

        # frame selection from h5 video infos
        self.start_index = start_index
        self.end_index = end_index
        self.image_nr = image_nr
        self.id_starter = id_starter

    # reading ImageJ Freehand ROIs from .zip file. Labels are expected to be freehand in ImaheJ.
    def read_imagej_rois(self):

        # roi_label path
        label_path = self.label_path

        # loading rois
        rois = read_roi_zip(label_path)

        # save to self
        self.rois = rois

    # read Ca2+ video (in h5 format) and save frames as image. To reduce image number part of video is saved as image.
    def video2image(self):

        # get data from self
        start_index = self.start_index
        end_index = self.end_index
        image_nr = self.image_nr
        id_starter = self.id_starter

        # get video
        video = self.video

        # get video shape (assume images are gray scale)
        n, x, y = video.shape

        # check end index
        if end_index == "all":
            end_index = n

        # check if end_index is numeric value
        assert isinstance(end_index, int)

        # select video based on the given parameters
        selected_video = video[start_index:end_index]

        # select frames randomly from selected_video
        selected_frames = selected_video[np.random.choice(
            a=selected_video.shape[0], size=image_nr, replace=False)]

        # write images in png format
        print("start saving images!")
        for i in tqdm(range(selected_frames.shape[0])):
            cv2.imwrite(os.path.join(self.output_path,
                        f"{id_starter + i}.png"), selected_frames[i])

    # convert to coco
    def convert2COCO(self):

        # get data from self
        rois = self.rois
        frame_path = self.output_path

        # get all image in directory
        image_list = os.listdir()

        # get annotations
        annotations = self.annotations

        # initialize dataset
        dataset = []

        # running in loop for all images
        print("start to write COCO style dataset!")
        for img in tqdm(image_list):

            # check it is ending with .png
            if img.endswith('.png'):

                # get direct image path
                img_path = os.path.join(frame_path, img)

                # read image to get its size
                img_data = []
                img_data = cv2.imread(img_path)

                # get height, width and channel numbers
                height, width, channels = img_data.shape

                # append images with all annotation info to dataset
                dataset.append({"file_name": img_path,
                                "image_id": img[:-4],
                                "height": height,
                                "width": width,
                                "annotations": annotations})

        # save to self
        self.dataset = dataset

    # tranform step (making conceptually similar to scikit-learn)
    def transform(self):

        # this function runs all helper function in order to make transformation happens.
        # 1. read imageJ rois
        print("step 1: reading imageJ ROIs")
        self.read_imagej_rois()

        # 2. loading video in h5 format
        print("step 2: loading video in h5 format")
        self.load_h5()

        # 3. change video to series of image and save them
        print("step 3: transfering video to images")
        self.video2image()

        # 4. create annotation dictionary
        print("step 4: creating annotation dictionary")
        self.ROIs2Annotation()

        # 5. converting to COCO style data set
        print("step 5: converting to COCO style dataset")
        self.convert2COCO()

        # return dataset
        return self.dataset

    # get annotation dictionary from rois
    def ROIs2Annotation(self):

        # get data from self
        rois = self.rois

        # fix parameters
        is_crowd = 0
        category_id = 0

        # create empty annotation array
        annotation = []

        # create annotation list
        for k, v in rois.items():

            # check if roi type is freehand (at the moment we are interested in freehand mode)
            if v['type'] == 'freehand':

                # clear x and y
                x = []
                y = []

                # load x and y
                x = v['x']
                y = v['y']

                # create xy
                xy = []
                for i in range(len(x)):
                    xy.append(x[i])
                    xy.append(y[i])

                # create a bbox
                bbox = [min(x), min(y), max(x) - min(x), max(y) - min(y)]

            # create segmentation dictionary
            annotation.append({"iscrowd": is_crowd,
                               "segmentation": xy,
                               "bbox": bbox,
                               "bbox_mode": BoxMode.XYWH_ABS,
                               "category_id": category_id})

            # save annotation to self
            self.annotation = annotation

    # loading h5 file
    def load_h5(self):

        # getting video path
        image_path = self.image_path

        # load h5 file
        File = h5py.File(image_path, 'r')
        for name in File.keys():
            print(name, '\n')

        # get correct key from user
        data_key = input("Please give data key from printed keys!")

        # check if the given key is in the keys
        assert data_key in File.keys()

        # load data based on the given key
        video = File[data_key]

        # put video in self
        self.video = video
