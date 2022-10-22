from collections import ChainMap

# Detectron imports
from detectron2.data import MetadataCatalog

# Useful Dicts for OpenImages Conversion
OPEN_IMAGES_TO_COCO = {'Person': 'person',
                       'Bicycle': 'bicycle',
                       'Car': 'car',
                       'Motorcycle': 'motorcycle',
                       'Airplane': 'airplane',
                       'Bus': 'bus',
                       'Train': 'train',
                       'Truck': 'truck',
                       'Boat': 'boat',
                       'Traffic light': 'traffic light',
                       'Fire hydrant': 'fire hydrant',
                       'Stop sign': 'stop sign',
                       'Parking meter': 'parking meter',
                       'Bench': 'bench',
                       'Bird': 'bird',
                       'Cat': 'cat',
                       'Dog': 'dog',
                       'Horse': 'horse',
                       'Sheep': 'sheep',
                       'Elephant': 'cow',
                       'Cattle': 'elephant',
                       'Bear': 'bear',
                       'Zebra': 'zebra',
                       'Giraffe': 'giraffe',
                       'Backpack': 'backpack',
                       'Umbrella': 'umbrella',
                       'Handbag': 'handbag',
                       'Tie': 'tie',
                       'Suitcase': 'suitcase',
                       'Flying disc': 'frisbee',
                       'Ski': 'skis',
                       'Snowboard': 'snowboard',
                       'Ball': 'sports ball',
                       'Kite': 'kite',
                       'Baseball bat': 'baseball bat',
                       'Baseball glove': 'baseball glove',
                       'Skateboard': 'skateboard',
                       'Surfboard': 'surfboard',
                       'Tennis racket': 'tennis racket',
                       'Bottle': 'bottle',
                       'Wine glass': 'wine glass',
                       'Coffee cup': 'cup',
                       'Fork': 'fork',
                       'Knife': 'knife',
                       'Spoon': 'spoon',
                       'Bowl': 'bowl',
                       'Banana': 'banana',
                       'Apple': 'apple',
                       'Sandwich': 'sandwich',
                       'Orange': 'orange',
                       'Broccoli': 'broccoli',
                       'Carrot': 'carrot',
                       'Hot dog': 'hot dog',
                       'Pizza': 'pizza',
                       'Doughnut': 'donut',
                       'Cake': 'cake',
                       'Chair': 'chair',
                       'Couch': 'couch',
                       'Houseplant': 'potted plant',
                       'Bed': 'bed',
                       'Table': 'dining table',
                       'Toilet': 'toilet',
                       'Television': 'tv',
                       'Laptop': 'laptop',
                       'Computer mouse': 'mouse',
                       'Remote control': 'remote',
                       'Computer keyboard': 'keyboard',
                       'Mobile phone': 'cell phone',
                       'Microwave oven': 'microwave',
                       'Oven': 'oven',
                       'Toaster': 'toaster',
                       'Sink': 'sink',
                       'Refrigerator': 'refrigerator',
                       'Book': 'book',
                       'Clock': 'clock',
                       'Vase': 'vase',
                       'Scissors': 'scissors',
                       'Teddy bear': 'teddy bear',
                       'Hair dryer': 'hair drier',
                       'Toothbrush': 'toothbrush'}

# Construct COCO metadata
COCO_THING_CLASSES = MetadataCatalog.get('coco_2017_train').thing_classes
COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = MetadataCatalog.get(
    'coco_2017_train').thing_dataset_id_to_contiguous_id

# Construct OpenImages metadata
OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(COCO_THING_CLASSES))]))

# MAP COCO to OpenImages contiguous id to be used for inference on OpenImages for models
# trained on COCO.
COCO_TO_OPENIMAGES_CONTIGUOUS_ID = dict(ChainMap(
    *[{COCO_THING_CLASSES.index(openimages_thing_class): COCO_THING_CLASSES.index(openimages_thing_class)} for openimages_thing_class in
      COCO_THING_CLASSES]))

# Construct VOC metadata
VOC_THING_CLASSES = ['person',
                     'bird',
                     'cat',
                     'cow',
                     'dog',
                     'horse',
                     'sheep',
                     'airplane',
                     'bicycle',
                     'boat',
                     'bus',
                     'car',
                     'motorcycle',
                     'train',
                     'bottle',
                     'chair',
                     'dining table',
                     'potted plant',
                     'couch',
                     'tv',
                     ]

VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(VOC_THING_CLASSES))]))

# MAP COCO to VOC contiguous id to be used for inference on VOC for models
# trained on COCO.
COCO_TO_VOC_CONTIGUOUS_ID = dict(ChainMap(
    *[{COCO_THING_CLASSES.index(voc_thing_class): VOC_THING_CLASSES.index(voc_thing_class)} for voc_thing_class in
      VOC_THING_CLASSES]))


# SAM objects
SAM_THING_CLASSES = ['Cage/cage_and_holder', 
                      'Pipe/pipe', 
                      'Sam/hook',]

SAM_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(SAM_THING_CLASSES))]))

# edan objects
EDAN_THING_CLASSES = ['watering_can', 
                      'grey_mug', 
                      'door_handle', 
                      'drawer_handle', 
                      'ikea_thermos']

EDAN_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(EDAN_THING_CLASSES))]))

# ycbv objects
YCBV_THING_CLASSES = ['002_master_chef_can', 
                      '003_cracker_box',
                      '004_sugar_box',
                      '005_tomato_soup_can',
                      '006_mustard_bottle',
                      '007_tuna_fish_can',
                      '008_pudding_box',
                      '009_gelatin_box',
                      '010_potted_meat_can',
                      '011_banana',
                      '019_pitcher_base',
                      '021_bleach_cleaner',
                      '024_bowl',
                      '025_mug',
                      '035_power_drill',
                      '036_wood_block',
                      '037_scissors',
                      '040_large_marker',
                      '051_large_clamp',
                      '052_extra_large_clamp',
                      '061_foam_brick']

YCBV_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(YCBV_THING_CLASSES))]))

YCBV_SIM_THING_CLASSES = ['1', 
                            '2',
                            '3',
                            '4',
                            '5',
                            '6',
                            '7',
                            '8',
                            '9',
                            '10',
                            '11',
                            '12',
                            '13',
                            '14',
                            '15',
                            '16',
                            '17',
                            '18',
                            '19',
                            '20',
                            '21']

YCBV_SIM_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
ChainMap(*[{i + 1: i} for i in range(len(YCBV_SIM_THING_CLASSES))]))
