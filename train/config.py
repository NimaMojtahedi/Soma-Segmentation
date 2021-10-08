# this python file is handeling model configuration (at the moment we start simple config mainly fixed)

# library
from detectron2.config import get_cfg
from detectron2 import model_zoo


# custom function to change Detectron2 configuration
def configuration(num_classes, train_output_path, min_image_size, image_per_batch, max_iter, model_weights=False, validation=False):

    # setup configuration file (at this stage very simple setup)

    # initialize configuration file
    cfg = get_cfg()

    # get default parameters of Instance Segmentation  with mask_rcnn_r_50_FPN
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    if model_weights:
        cfg.MODEL.WEIGHTS = model_weights
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Model parameters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    #cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"

    # SOLVER parameters
    #cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = max_iter
    #cfg.SOLVER.STEPS = (20, 10000, 20000)
    #cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = image_per_batch

    # Test parameters
    cfg.TEST.DETECTIONS_PER_IMAGE = 25

    # INPUT parameters
    cfg.INPUT.MIN_SIZE_TRAIN = (min_image_size,)
    cfg.INPUT.MIN_SIZE_TEST = (min_image_size,)

    # DATASETS
    if validation:
        cfg.DATASETS.TEST = ('val',)
    else:
        cfg.DATASETS.TEST = ()
    cfg.DATASETS.TRAIN = ('train',)

    # output location
    cfg.OUTPUT_DIR = train_output_path

    return cfg
