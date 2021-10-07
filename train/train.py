# this file is handeling train section

# libraries
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from data_loader import load_dataset
from config import detectron2_config
from data_loader import detectron2_register


def run_train(data_path):

    # loading json/train file
    data = load_dataset(
        data_path=data_path)

    # register dataset in Detectron2
    metadata = detectron2_register(
        datasetfunct=lambda path: load_dataset(path), classes=["soma"], dataset_name="train")

    # load configuration file
    cfg = detectron2_config()

    # start training with defined parameters in cfg file
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
