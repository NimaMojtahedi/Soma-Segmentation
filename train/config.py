# this python file is handeling model configuration (at the moment we start simple config mainly fixed)

# library
from detectron2.config import get_cfg
from detectron2 import model_zoo


# custom function to change Detectron2 configuration
def detectron2_config(num_classes, output_path):

    # initialize configuration file
    cfg = get_cfg()

    # get default parameters of Instance Segmentation  with mask_rcnn_r_50_FPN
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Model parameters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"

    # SOLVER parameters
    #cfg.SOLVER.BASE_LR = 0.0002
    #cfg.SOLVER.MAX_ITER = 40000
    #cfg.SOLVER.STEPS = (20, 10000, 20000)
    #cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2

    # Test parameters
    cfg.TEST.DETECTIONS_PER_IMAGE = 25

    # INPUT parameters
    cfg.INPUT.MIN_SIZE_TRAIN = (250,)
    cfg.INPUT.MIN_SIZE_TEST = (250,)

    # DATASETS
    cfg.DATASETS.TEST = ('val',)
    cfg.DATASETS.TRAIN = ('train',)

    # output location
    cfg.OUTPUT_DIR = output_path

    return cfg
