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
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import ImageJ2COCO
from config import configuration
import pdb

if __name__ == "__main__":

    # evaluation inference
    eval_inf = True

    # register train
    register_coco_instances("train", {}, "your_path/val.json",
                            "your_path/publaynet/val")
    register_coco_instances("val", {}, "your_path/publaynet/val.json",
                            "your_path/publaynet/val")

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

    # start training with custom dataloader and custom data augmentation step

    class CustomTrainer(DefaultTrainer):

        @classmethod
        def build_train_loader(cls, cfg):
            dataloader = build_detection_train_loader(cfg,
                                                      mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                                                          T.Resize((800, 800)),
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

    # check if evaluation is required
    if eval_inf:
        evaluator = COCOEvaluator(
            "val", cfg, False, output_dir="C:/Users/admin/Desktop/test/out2")
        val_loader = build_detection_test_loader(cfg, "val")
        inference_on_dataset(trainer.model, val_loader, evaluator)
