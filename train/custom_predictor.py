# running prediction in many images in loop

# libraries
import os
from config import configuration
from segmentation_predictor import predict_img
import pandas as pd

# save result's image
save_img = False

# get dir path to images
imgs_path = ""

# get images info
imgs = os.listdir(imgs_path)

# model weights (after training)
model_weights_path = ""

# path for saving final results
save_path = ""

# get configuration file same as used during training
cfg = configuration(num_classes=1,
                    train_output_path="C:/Users/admin/Desktop/COCO test2/out",
                    min_image_size=240,
                    image_per_batch=1,
                    max_iter=150,
                    model_weights=model_weights_path,
                    validation=False)

# initialize main dataframe
df = pd.DataFrame()

# starting looping results
for img in imgs:
    df = df.append(predict_img(cfg=cfg, img_path=img, save_path=save_path,
                               img_save=save_img, df_save=False, score_thresh=0.7), ignore_index=True)

# save main dataframe
print("saving data frame!")
df.to_csv(os.path.join(save_path, "main_results") + ".csv", index=False)
