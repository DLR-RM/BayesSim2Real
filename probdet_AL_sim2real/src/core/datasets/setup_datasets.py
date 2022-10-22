import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import src.core.datasets.metadata as metadata

def setup_all_datasets(dataset_dir=None, image_root_corruption_prefix=None):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_edan_coco_data()
    setup_ycbv_coco_data()
    setup_SAM_coco_data()

def setup_SAM_coco_data():
    SAM_ROOT = "./Datasets/SAM_objects/generated_datasets/merged_train_syn_SAM"

    sam_img_dir = f"{SAM_ROOT}/images"
    sam_anno_dir = f"{SAM_ROOT}/annos"
    edan_sim_json_annotations = os.path.join(
        sam_anno_dir, 'annotations_filtered.json')
    register_coco_instances(
        "sam_sim",
        {},
        edan_sim_json_annotations,
        sam_img_dir)
    MetadataCatalog.get(
        "sam_sim").thing_classes = metadata.SAM_THING_CLASSES
    MetadataCatalog.get(
        "sam_sim").thing_dataset_id_to_contiguous_id = metadata.SAM_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_ycbv_coco_data():
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    YCBV_ANN_ROOT = "./ycbv_data/ycbv_test_train_real_sim_data"
    # YCBV_IMG_ROOT = "/bop/original/ycbv/"

    ycbv_sim_img_dir = f"{YCBV_ANN_ROOT}/ycbv_train_pbr_annotations/merged/annos"
    ycbv_sim_json_annotations = os.path.join(f"{YCBV_ANN_ROOT}/ycbv_train_pbr_annotations/merged/", 'annos/annotations.json')
    register_coco_instances(
        "ycbv_sim",
        {},
        ycbv_sim_json_annotations,
        ycbv_sim_img_dir)
    # MetadataCatalog.get(
    #    "ycbv_sim").thing_classes = metadata.YCBV_SIM_THING_CLASSES
    # MetadataCatalog.get(
    #     "ycbv_sim").thing_dataset_id_to_contiguous_id = metadata.YCBV_SIM_THING_DATASET_ID_TO_CONTIGUOUS_ID

    ycbv_real_train_dir = f"{YCBV_ANN_ROOT}/ycbv_train_real_annotations/merged/annos/"
    ycbv_real_train_img_dir = f"{YCBV_ANN_ROOT}/ycbv_train_real_annotations/merged/annos/"
    ycbv_real_train_annotations = os.path.join(ycbv_real_train_dir, 'annotations_split_test.json')
    register_coco_instances(
        "ycbv_real_val",
        {},
        ycbv_real_train_annotations,
        ycbv_real_train_img_dir)
    MetadataCatalog.get(
        "ycbv_real_val").thing_classes = metadata.YCBV_THING_CLASSES
    MetadataCatalog.get(
        "ycbv_real_val").thing_dataset_id_to_contiguous_id = metadata.YCBV_THING_DATASET_ID_TO_CONTIGUOUS_ID

    ycbv_real_train_dir = f"{YCBV_ANN_ROOT}/ycbv_train_real_annotations/merged/annos/"
    ycbv_real_train_img_dir = f"{YCBV_ANN_ROOT}/ycbv_train_real_annotations/merged/annos/"
    ycbv_real_train_annotations = os.path.join(ycbv_real_train_dir, 'annotations_split_train.json')
    register_coco_instances(
        "ycbv_real_train",
        {},
        ycbv_real_train_annotations,
        ycbv_real_train_img_dir)
    MetadataCatalog.get(
        "ycbv_real_train").thing_classes = metadata.YCBV_THING_CLASSES
    MetadataCatalog.get(
        "ycbv_real_train").thing_dataset_id_to_contiguous_id = metadata.YCBV_THING_DATASET_ID_TO_CONTIGUOUS_ID

    ycbv_real_test_dir = f"{YCBV_ANN_ROOT}/ycbv_test_annotations/merged/annos/"
    ycbv_real_test_img_dir = f"{YCBV_ANN_ROOT}/ycbv_test_annotations/merged/annos/"
    ycbv_real_test_annotations = os.path.join(ycbv_real_test_dir, 'annotations.json')
    register_coco_instances(
        "ycbv_real_test",
        {},
        ycbv_real_test_annotations,
        ycbv_real_test_img_dir)
    MetadataCatalog.get(
        "ycbv_real_test").thing_classes = metadata.YCBV_THING_CLASSES
    MetadataCatalog.get(
        "ycbv_real_test").thing_dataset_id_to_contiguous_id = metadata.YCBV_THING_DATASET_ID_TO_CONTIGUOUS_ID

    '''
    ycbv_real_test_all_dir = f"{YCBV_ANN_ROOT}/ycbv_test_annotations/merged_all/annos/"
    ycbv_real_test_all_img_dir = f"{YCBV_IMG_ROOT}/test/"
    ycbv_real_test_all_annotations = os.path.join(ycbv_real_test_all_dir, 'annotations.json')
    register_coco_instances(
        "ycbv_real_test_all",
        {},
        ycbv_real_test_all_annotations,
        ycbv_real_test_all_img_dir)
    MetadataCatalog.get(
        "ycbv_real_test_all").thing_classes = metadata.YCBV_THING_CLASSES
    MetadataCatalog.get(
        "ycbv_real_test_all").thing_dataset_id_to_contiguous_id = metadata.YCBV_THING_DATASET_ID_TO_CONTIGUOUS_ID
    '''

def setup_edan_coco_data():
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    EDAN_IMG_ROOT = "./EDAN_DATA/edan_sim2real"

    edan_sim_dir = f"{EDAN_IMG_ROOT}/edan_objects/merged/images"
    edan_sim_json_annotations = os.path.join(
        edan_sim_dir, 'annotations_mapped.json')
    register_coco_instances(
        "edan_sim",
        {},
        edan_sim_json_annotations,
        edan_sim_dir)
    MetadataCatalog.get(
        "edan_sim").thing_classes = metadata.EDAN_THING_CLASSES
    MetadataCatalog.get(
        "edan_sim").thing_dataset_id_to_contiguous_id = metadata.EDAN_THING_DATASET_ID_TO_CONTIGUOUS_ID

    edan_real_train_dir = f"{EDAN_IMG_ROOT}/edan_real_coco/images"
    edan_real_train_annotations = os.path.join(edan_real_train_dir, 'annotations_mapped_split_train.json')
    register_coco_instances(
        "edan_real_train",
        {},
        edan_real_train_annotations,
        edan_real_train_dir)
    MetadataCatalog.get(
        "edan_real_train").thing_classes = metadata.EDAN_THING_CLASSES
    MetadataCatalog.get(
        "edan_real_train").thing_dataset_id_to_contiguous_id = metadata.EDAN_THING_DATASET_ID_TO_CONTIGUOUS_ID

    edan_real_val_dir = f"{EDAN_IMG_ROOT}/edan_real_coco/images"
    edan_real_val_annotations = os.path.join(edan_real_val_dir, 'annotations_mapped_split_val.json')
    register_coco_instances(
        "edan_real_val",
        {},
        edan_real_val_annotations,
        edan_real_val_dir)
    MetadataCatalog.get(
        "edan_real_val").thing_classes = metadata.EDAN_THING_CLASSES
    MetadataCatalog.get(
        "edan_real_val").thing_dataset_id_to_contiguous_id = metadata.EDAN_THING_DATASET_ID_TO_CONTIGUOUS_ID

    edan_real_test_dir = f"{EDAN_IMG_ROOT}/edan_real_coco/images"
    edan_real_test_annotations = os.path.join(edan_real_test_dir, 'annotations_mapped_split_test.json')
    register_coco_instances(
        "edan_real_test",
        {},
        edan_real_test_annotations,
        edan_real_test_dir)
    MetadataCatalog.get(
        "edan_real_test").thing_classes = metadata.EDAN_THING_CLASSES
    MetadataCatalog.get(
        "edan_real_test").thing_dataset_id_to_contiguous_id = metadata.EDAN_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_dataset(dataset_dir, image_root_corruption_prefix=None):
    """
    sets up coco dataset following detectron2 coco instance format. Required to not have flexibility on where the dataset
    files can be.
    """
    train_image_dir = os.path.join(dataset_dir, 'train2017')

    if image_root_corruption_prefix is not None:
        test_image_dir = os.path.join(
            dataset_dir, 'val2017' + image_root_corruption_prefix)
    else:
        test_image_dir = os.path.join(dataset_dir, 'val2017')

    train_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017.json')

    register_coco_instances(
        "coco_2017_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_2017_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_classes = metadata.COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_dataset_id_to_contiguous_id = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

