# This file provides utility functions

# Necessary packages
from read_roi import read_roi_zip


# convert ImageJ labels to coco format dataset
class ImageJ2COCO:

    # initialize the class
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    # reading ImageJ Freehand ROIs from .zip file. Labels are expected to be freehand in ImaheJ.
    def read_imagej_rois(self):

        # roi_label path
        label_path = self.label_path

        # loading rois
        rois = read_roi_zip(label_path)

        # save to self
        self.rois = rois

    # read Ca2+ video and save frames as image
    def video2image(self):
        pass

    # convert to coco
    def convert2COCO(self):
        pass
