# this file is handeling train section
import os
# libraries
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

from utils import ImageJ2COCO
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import random
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import ImageJ2COCO
from config import configuration


if __name__ == "__main__":

    # training using custom trainer or default trainer
    default_trainer = False

    # dataloader for train dataset (this custom written data loader is able to load multple datasets from different location)
    img2coco_train = ImageJ2COCO(image_path=["G:/Data & Analysis/150802_p3.5_gcamp6/Data/150802_p3.5_gcamp6 H5/150802_a3_1h40min.h5",
                                             "G:/Data & Analysis/150802_p3.5_gcamp6/Data/150802_p3.5_gcamp6 H5/150802_a3_1h40min.h5"],
                                 label_path=["G:/Data & Analysis/150802_p3.5_gcamp6/Analysis/ROIS and Inside Activities/RoiSetFull.zip",
                                             "G:/Data & Analysis/150802_p3.5_gcamp6/Analysis/ROIS and Inside Activities/RoiSetFull.zip"],
                                 output_path="C:/Users/admin/Desktop/test",
                                 start_index=[10000, 12000],
                                 end_index=[12000, 14000],
                                 image_nr=[1400, 1500],
                                 id_starter=[1, 2000],
                                 min_intensity=[100, 100],
                                 max_intensity=[4000, 3000],
                                 key=["GroupHierarchy.Groups.Datasets",
                                      "GroupHierarchy.Groups.Datasets"])

    # dataloader for validation dataset (this custom written data loader is able to load multple datasets from different location)
    img2coco_val = ImageJ2COCO(image_path=["G:/Data & Analysis/150802_p3.5_gcamp6/Data/150802_p3.5_gcamp6 H5/150802_a3_1h40min.h5",
                                           "G:/Data & Analysis/150802_p3.5_gcamp6/Data/150802_p3.5_gcamp6 H5/150802_a3_1h40min.h5"],
                               label_path=["G:/Data & Analysis/150802_p3.5_gcamp6/Analysis/ROIS and Inside Activities/RoiSetFull.zip",
                                           "G:/Data & Analysis/150802_p3.5_gcamp6/Analysis/ROIS and Inside Activities/RoiSetFull.zip"],
                               output_path="C:/Users/admin/Desktop/test",
                               start_index=[1000, 1200],
                               end_index=[20000, 40000],
                               image_nr=[800, 800],
                               id_starter=[3000, 4000],
                               min_intensity=[100, 100],
                               max_intensity=[4000, 3000],
                               key=["GroupHierarchy.Groups.Datasets",
                               "GroupHierarchy.Groups.Datasets"])

    # register train data set
    DatasetCatalog.register("train", img2coco_train.transform)
    MetadataCatalog.get("train").set(
        thing_classes=["soma"])  # define classes as well

    # register validation data set
    DatasetCatalog.register("val", img2coco_val.transform)
    MetadataCatalog.get("val").set(
        thing_classes=["soma"])  # define classes as well

    metadata = MetadataCatalog.get("train")

    # configuration file
    cfg = configuration(num_classes=1,
                        train_output_path="C:/Users/admin/Desktop/test/out",
                        min_image_size=240,
                        image_per_batch=1,
                        max_iter=150,
                        model_weights="C:/Users/admin/Desktop/test/out/model_final.pth",
                        validation=True)

    if default_trainer:

        # start training
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    else:
        # start training with custom dataloader and custom data augmentation step
        class CustomTrainer(DefaultTrainer):

            @classmethod
            def build_train_loader(cls, cfg):
                dataloader = build_detection_train_loader(cfg,
                                                          mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                                                              T.Resize(
                                                                  (800, 800)),
                                                              T.RandomBrightness(
                                                                  intensity_min=0.5, intensity_max=2),
                                                              T.RandomContrast(
                                                                  intensity_min=0.5, intensity_max=2),
                                                              T.RandomCrop(
                                                                  crop_type="relative", crop_size=(0.8, 0.8)),
                                                              T.RandomFlip()
                                                          ]))
                return dataloader

        # https://www.kaggle.com/dhiiyaur/detectron-2-compare-models-augmentation
        # first time install shapely

        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
