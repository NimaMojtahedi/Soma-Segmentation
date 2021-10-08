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
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import ImageJ2COCO
from config import configuration


if __name__ == "__main__":

    # getting imagej to coco from utils
    img2coco = ImageJ2COCO(image_path=["G:/Data & Analysis/150802_p3.5_gcamp6/Data/150802_p3.5_gcamp6 H5/150802_a3_1h40min.h5",
                                       "G:/Data & Analysis/150802_p3.5_gcamp6/Data/150802_p3.5_gcamp6 H5/150802_a3_1h40min.h5"],
                           label_path=["G:/Data & Analysis/150802_p3.5_gcamp6/Analysis/ROIS and Inside Activities/RoiSetFull.zip",
                                       "G:/Data & Analysis/150802_p3.5_gcamp6/Analysis/ROIS and Inside Activities/RoiSetFull.zip"],
                           output_path="C:/Users/admin/Desktop/COCO test2",
                           start_index=[10000, 12000],
                           end_index=[12000, 14000],
                           image_nr=[40, 50],
                           id_starter=[1, 100],
                           min_intensity=[100, 100],
                           max_intensity=[4000, 3000],
                           key=["GroupHierarchy.Groups.Datasets",
                                "GroupHierarchy.Groups.Datasets"])

    # register train
    DatasetCatalog.register("train", img2coco.transform)
    MetadataCatalog.get("train").set(thing_classes=["Soma"])
    metadata = MetadataCatalog.get("train")

    # configuration file
    cfg = configuration(num_classes=1,
                        train_output_path="C:/Users/admin/Desktop/COCO test2/out",
                        min_image_size=240,
                        image_per_batch=1,
                        max_iter=150,
                        model_weights=False,
                        validation=False)

    # start training

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
