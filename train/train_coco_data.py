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
import pdb

if __name__ == "__main__":

    # getting imagej to coco from utils

    # register train
    register_coco_instances("train", {}, "I:/Sina/Medical report segmentation/publaynet/val.json",
                            "I:/Sina/Medical report segmentation/publaynet/val")
    register_coco_instances("val", {}, "I:/Sina/Medical report segmentation/publaynet/val.json",
                            "I:/Sina/Medical report segmentation/publaynet/val")

    # adding class name
    MetadataCatalog.get("val").set(
        thing_classes=["title", "text", "figure", "table", "list"])
    MetadataCatalog.get("train").set(
        thing_classes=["title", "text", "figure", "table", "list"])

    # configuration file
    cfg = configuration(num_classes=5,
                        train_output_path="C:/Users/admin/Desktop/test/out",
                        min_image_size=240,
                        image_per_batch=1,
                        max_iter=150,
                        model_weights=False, validation=True)

    # start training

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
