from detectron2.data.datasets import register_coco_instances
import src.core.datasets.metadata as metadata
from detectron2.data import MetadataCatalog

def register_new_edan_coco_data(dataset_name, anno_path, image_folder):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    # edan_real_train_annotations = os.path.join(
    #     edan_real_train_dir, 'annotations_mapped_split_train.json')
    register_coco_instances(
        dataset_name,
        {},
        anno_path,
        image_folder)
    MetadataCatalog.get(
        dataset_name).thing_classes = metadata.EDAN_THING_CLASSES
    MetadataCatalog.get(
        dataset_name).thing_dataset_id_to_contiguous_id = metadata.EDAN_THING_DATASET_ID_TO_CONTIGUOUS_ID

def register_new_ycbv_coco_data(dataset_name, anno_path, image_folder):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    # edan_real_train_annotations = os.path.join(
    #     edan_real_train_dir, 'annotations_mapped_split_train.json')
    register_coco_instances(
        dataset_name,
        {},
        anno_path,
        image_folder)
    MetadataCatalog.get(
        dataset_name).thing_classes = metadata.YCBV_THING_CLASSES
    MetadataCatalog.get(
        dataset_name).thing_dataset_id_to_contiguous_id = metadata.YCBV_THING_DATASET_ID_TO_CONTIGUOUS_ID
