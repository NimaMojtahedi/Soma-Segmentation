# In detectron2 to load data from json format we need to register it in Detectron2 dataset catalog.

# import librairies
import json
from detectron2.data import MetadataCatalog, DatasetCatalog


# function to load json file
def load_dataset(data_path):

    with open(data_path, 'r') as f:
        dataset = json.load(f)


# register and define classes
def detectron2_register(dataset_func, classes, dataset_name="train"):

    DatasetCatalog.register(dataset_name, dataset_func)
    MetadataCatalog.get(dataset_name).set(thing_classes=classes)

    return MetadataCatalog.get(dataset_name)
