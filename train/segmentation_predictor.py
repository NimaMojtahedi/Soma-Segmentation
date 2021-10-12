# this file helps to predict segmentation info on single img

# libraries
from detectron2.engine import DefaultPredictor
import pandas as pd
import random
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt
import os
import pdb


def predict_img(cfg, img_path, save_path, img_save=False, df_save=False, score_thresh=0.7):

    # path to the model we just trained
    #cfg.MODEL.WEIGHTS = model_weight_path

    # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    # prepare default predictor using custom configuration file
    predictor = DefaultPredictor(cfg)

    # read image
    im = cv2.imread(img_path)

    # get path info
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    file_ext = os.path.splitext(os.path.basename(img_path))[1]
    dir_name = os.path.dirname(img_path)

    # predict bbox and segmentation
    outputs = predictor(im)

    # write results to dataframe
    df = pd.DataFrame()
    df = df.append({"image_path": img_path,
                    'image_name': file_name,
                    "masks": outputs["instances"].pred_masks.to("cpu").numpy(),
                    "scores": outputs["instances"].scores.to("cpu").numpy(),
                    "pred_classes": outputs["instances"].pred_classes.to("cpu").numpy(),
                    "pred_boxes": outputs["instances"].pred_boxes.tensor.to("cpu").numpy()}, ignore_index=True)

    # saving results in csv format
    if df_save:
        df.to_csv(os.path.join(save_path, file_name) + ".csv", index=False)

    # saving image with overlaid segmentation
    if img_save:

        # convert image with detectron2 visualizer
        v = Visualizer(im[:, :, ::-1],
                       metadata=None,
                       scale=3,
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)

        # saving images with segmentation and bbox info on top
        plt.imsave(os.path.join(save_path, file_name) + ".png", img)

    return df
