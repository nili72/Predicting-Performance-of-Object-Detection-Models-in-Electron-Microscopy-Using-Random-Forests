from pprint import pprint
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer

from detectron2.structures import BoxMode
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import ColorMode

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


def get_defect_anno_dict_train(train_annotations_path, train_dataset_path):
    print('GETTING DEFECT TRAIN ANNOTATION DICT')
    anno_dict_list = list()
    with open(train_annotations_path, 'r') as f:
        n_defects = 0  # count the total number of defects in training images
        anno = json.load(f)
        basic_image_data = anno['images']
        num_train_imgs = len(basic_image_data)

        annotation_image_data = anno['annotations']
        category_image_data = anno['categories']

        # print(category_image_data)

        def modify_bbox(initial_bbox):
            bbox1 = initial_bbox[0]
            bbox2 = initial_bbox[1]
            bbox3 = bbox1 + initial_bbox[2]
            bbox4 = bbox2 + initial_bbox[3]
            bbox = [bbox1, bbox2, bbox3, bbox4]
            return bbox

        # Loop over image number and find objects in each image
        for img in range(num_train_imgs):
            filenames = list()
            heights = list()
            widths = list()
            iscrowds = list()
            bboxes = list()
            segmentations = list()
            bbox_modes = list()
            category_ids = list()
            # Original data was separated by object, not by image
            for obj in annotation_image_data:
                if obj['image_id'] == img:
                    # Here- need path to file
                    # filenames.append(os.path.join('defect_detection/train', basic_image_data[img]['file_name']))
                    filenames.append(basic_image_data[img]['file_name'])
                    heights.append(basic_image_data[img]['height'])
                    widths.append(basic_image_data[img]['width'])
                    # pprint(obj)
                    # print('On image')
                    # print(obj['image_id'])
                    iscrowds.append(obj['iscrowd'])

                    bboxes.append(modify_bbox(obj['bbox']))  # If doing XYXY_ABS bboxes

                    # category_ids.append(obj['category_id']) # What I had before
                    # category_ids.append(obj['category_id']-1) # Start at 0 because naming is done with list

                    # Just have equal 0 for now- 1 object
                    # category_ids.append(0)

                    # For Mask-RNN annotations. Try 3 categories. If 2 then dot, if 1 then 111, else 100
                    #if obj['category_id'] == 2:  # bd defect
                    #    category_ids.append(0)
                    #elif obj['category_id'] == 1:  # 111 defect
                    #    category_ids.append(1)
                    #else:
                    #    category_ids.append(2)  # 100 defect, combining 100_1 and 100_2 into 100
                    if obj['category_id'] == 3:  # bd defect
                        category_ids.append(0)
                    elif obj['category_id'] == 2:  # 111 defect
                        category_ids.append(1)
                    else:
                        category_ids.append(2)  # 100 defect, combining 100_1 and 100_2 into 100

                    segmentations.append(obj['segmentation'])
                    bbox_modes.append(BoxMode.XYXY_ABS)  # what I had before to follow balloon dataset
                    # bbox_modes.append(BoxMode.XYWH_ABS) #needed if don't change bboxes

                    n_defects += 1

            anno_dict = dict()
            anno_dict['annotations'] = list()
            for iscrowd, bbox, category_id, segmentation, bbox_mode in zip(iscrowds, bboxes, category_ids,
                                                                           segmentations, bbox_modes):
                anno_dict['annotations'].append(
                    {'iscrowd': iscrowd, 'bbox': bbox, 'category_id': category_id, 'segmentation': segmentation,
                     'bbox_mode': bbox_mode})
            anno_dict['file_name'] = os.path.join(train_dataset_path, basic_image_data[img]['file_name'])
            # anno_dict['file_name'] = basic_image_data[img]['file_name'] # what I had before
            anno_dict['height'] = basic_image_data[img]['height']
            anno_dict['width'] = basic_image_data[img]['width']
            # pprint(anno_dict)
            # print('Number of object in image...')
            # print(len(anno_dict['annotations']))
            anno_dict_list.append(anno_dict)

    print('Number of defects in training images')
    print(n_defects)
    return anno_dict_list

def get_defect_anno_dict_val(test_annotations_path, test_dataset_path):
    print('GETTING DEFECT VAL ANNOTATION DICT')
    anno_dict_list = list()
    with open(test_annotations_path, 'r') as f:
        n_defects = 0  # count the total number of defects in val images
        anno = json.load(f)
        basic_image_data = anno['images']
        num_test_imgs = len(basic_image_data)

        annotation_image_data = anno['annotations']
        category_image_data = anno['categories']

        # print(category_image_data)

        def modify_bbox(initial_bbox):
            bbox1 = initial_bbox[0]
            bbox2 = initial_bbox[1]
            bbox3 = bbox1 + initial_bbox[2]
            bbox4 = bbox2 + initial_bbox[3]
            bbox = [bbox1, bbox2, bbox3, bbox4]
            return bbox

        # Loop over image number and find objects in each image
        for img in range(num_test_imgs):
            filenames = list()
            heights = list()
            widths = list()
            iscrowds = list()
            bboxes = list()
            segmentations = list()
            bbox_modes = list()
            category_ids = list()
            # Original data was separated by object, not by image
            for obj in annotation_image_data:
                if obj['image_id'] == img:
                    # Here- need path to file
                    # filenames.append(os.path.join('defect_detection/val', basic_image_data[img]['file_name']))
                    filenames.append(basic_image_data[img]['file_name'])
                    heights.append(basic_image_data[img]['height'])
                    widths.append(basic_image_data[img]['width'])
                    # pprint(obj)
                    # print('On image')
                    # print(obj['image_id'])
                    iscrowds.append(obj['iscrowd'])

                    bboxes.append(modify_bbox(obj['bbox']))  # If doing XYXY_ABS bboxes
                    # bboxes.append(obj['bbox'])

                    # category_ids.append(obj['category_id']) # What I had before
                    # category_ids.append(obj['category_id']-1) # Start at 0 because naming is done with list

                    # Just have equal 0 for now- 1 object
                    # category_ids.append(0)

                    # Try 3 categories. For Mask R-CNN annotations, If 2 then dot, if 1 then 111, else 100
                    #if obj['category_id'] == 2:  # bd defect
                    #    category_ids.append(0)
                    #elif obj['category_id'] == 1:  # 111 defect
                    #    category_ids.append(1)
                    #else:
                    #    category_ids.append(2)  # 100 defect
                    if obj['category_id'] == 3:  # bd defect
                        category_ids.append(0)
                    elif obj['category_id'] == 2:  # 111 defect
                        category_ids.append(1)
                    else:
                        category_ids.append(2)  # 100 defect, combining 100_1 and 100_2 into 100

                    segmentations.append(obj['segmentation'])
                    bbox_modes.append(BoxMode.XYXY_ABS)  # what I had before to follow balloon dataset

                    n_defects += 1

            anno_dict = dict()
            anno_dict['annotations'] = list()
            for iscrowd, bbox, category_id, segmentation, bbox_mode in zip(iscrowds, bboxes, category_ids,
                                                                           segmentations, bbox_modes):
                anno_dict['annotations'].append(
                    {'iscrowd': iscrowd, 'bbox': bbox, 'category_id': category_id, 'segmentation': segmentation,
                     'bbox_mode': bbox_mode})
            anno_dict['file_name'] = os.path.join(test_dataset_path, basic_image_data[img]['file_name'])
            # anno_dict['file_name'] = basic_image_data[img]['file_name'] # what I had before
            anno_dict['height'] = basic_image_data[img]['height']
            anno_dict['width'] = basic_image_data[img]['width']
            # pprint(anno_dict)
            # print('Number of object in image...')
            # print(len(anno_dict['annotations']))
            anno_dict_list.append(anno_dict)
    print('Number of defects in val images')
    print(n_defects)
    return anno_dict_list

def get_defect_metadata(train_dataset_path, test_dataset_path, train_annotations_path, test_annotations_path):
    # Attach the image metadata for class labels to the images
    DatasetCatalog.register(train_dataset_path, lambda : get_defect_anno_dict_train(train_annotations_path, train_dataset_path))
    MetadataCatalog.get(train_dataset_path).set(thing_classes=["bdot", "111", "100"], thing_colors=[(0,0,255), (255,0,0), (255,255,0)]) # trying three class labels
    #MetadataCatalog.get(train_dataset_path).set(thing_classes=["111", "100", "bd"], thing_colors=[(255,0,0), (255,255,0), (0,0,255)]) # trying three class labels
    defect_metadata = MetadataCatalog.get(train_dataset_path)
    DatasetCatalog.register(test_dataset_path, lambda : get_defect_anno_dict_val(test_annotations_path, test_dataset_path))
    MetadataCatalog.get(test_dataset_path).set(thing_classes=["bdot", "111", "100"], thing_colors=['blue', 'red', 'yellow']) # trying three class labels
    #MetadataCatalog.get(test_dataset_path).set(thing_classes=["111", "100", "bd"], thing_colors=[(255,0,0), (255,255,0), (0,0,255)]) # trying three class labels
    print('Defect metadata')
    pprint(defect_metadata.as_dict())
    return defect_metadata

def visualize_image(anno_dict_list):
    # Get random image and show it
    for anno_dict in random.sample(anno_dict_list, 1):
    #for anno_dict in anno_dict_list:
        # Just set as first image to be reproducible:
        anno_dict = anno_dict_list[0]
        print('Visualizing image')
        print(anno_dict["file_name"])
        img = cv2.imread(anno_dict["file_name"])
        print('Image shape')
        print(img.shape)

        # Assign color to each defect in the image
        assigned_colors_list = list()
        for defect in anno_dict['annotations']:
            id = defect['category_id']
            if id == 0: #bdot
                assigned_colors_list.append('b')
            elif id == 1: #111
                assigned_colors_list.append('r')
            else:
                assigned_colors_list.append('y')
        anno_dict['assigned_colors'] = assigned_colors_list

        cv2_imshow(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=defect_metadata, scale=0.7, instance_mode=ColorMode.SEGMENTATION)
        vis = visualizer.draw_dataset_dict(anno_dict)
        cv2_imshow(vis.get_image()[:, :, ::-1])
        return

def get_config_file(input_yaml):
    cfg = get_cfg()
    if input_yaml['mask_on'] == True:
        if input_yaml['cascade_maskrcnn'] == True:
            cfg.merge_from_file("/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/Misc/" + str(input_yaml['pretrained_model_name']))
        else:
            cfg.merge_from_file("/jet/home/rjacobs/detectron2_configs/configs/COCO-InstanceSegmentation/"+str(input_yaml['pretrained_model_name']))
    elif input_yaml['mask_on'] == False:
        cfg.merge_from_file("/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-Detection/" + str(input_yaml['pretrained_model_name']))
    else:
        # Assume doing Mask R-CNN
        cfg.merge_from_file("/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/"+str(input_yaml['pretrained_model_name']))

    # cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137849600/model_final_f10217.pkl"  # initialize from model zoo

    # Note that you can download the model weights from links provided on https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    model_weights_urls = {"cascade_mask_rcnn_R_50_FPN_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/Misc/model_weights/cascade_mask_rcnn_R_50_FPN_1x.pkl",
                          "cascade_mask_rcnn_R_50_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/Misc/model_weights/cascade_mask_rcnn_R_50_FPN_3x.pkl",

                        "mask_rcnn_R_50_C4_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_C4_1x_model_final_9243eb.pkl",
                        "mask_rcnn_R_50_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_C4_3x_model_final_4ce675.pkl",
                        "mask_rcnn_R_50_DC5_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_DC5_1x_model_final_4f86c3.pkl",
                        "mask_rcnn_R_50_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_DC5_3x_model_final_84107b.pkl",
                        "mask_rcnn_R_50_FPN_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_FPN_1x_model_final_a54504.pkl",
                        "mask_rcnn_R_50_FPN_3x": "/jet/home/rjacobs/detectron2_configs/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_FPN_3x_model_final_f10217.pkl",
                        "mask_rcnn_R_50_FPN_3x_balloon": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_50_FPN_3x_balloon_model_final.pth",
                        "mask_rcnn_R_101_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_101_C4_3x_model_final_a2914c.pkl",
                        "mask_rcnn_R_101_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_101_DC5_3x_model_final_0464b7.pkl",
                        "mask_rcnn_R_101_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_R_101_FPN_3x_model_final_a3ec72.pkl",
                        "mask_rcnn_X_101_32x8d_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/mask_rcnn_X_101_32x8d_FPN_3x_model_final_2d9806.pkl",

                          "faster_rcnn_R_50_C4_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_C4_1x_model_final_721ade.pkl",
                          "faster_rcnn_R_50_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_C4_3x_model_final_f97cb7.pkl",
                          "faster_rcnn_R_50_DC5_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_DC5_1x_model_final_51d356.pkl",
                          "faster_rcnn_R_50_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_DC5_3x_model_final_68d202.pkl",
                          "faster_rcnn_R_50_FPN_1x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_FPN_1x_model_final_b275ba.pkl",
                          "faster_rcnn_R_50_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_50_FPN_3x_model_final_280758.pkl",
                          "faster_rcnn_R_101_C4_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_101_C4_3x_model_final_298dad.pkl",
                          "faster_rcnn_R_101_DC5_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_101_DC5_3x_model_final_3e0943.pkl",
                          "faster_rcnn_R_101_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_R_101_FPN_3x_model_final_f6e8b1.pkl",
                          "faster_rcnn_X_101_32x8d_FPN_3x": "/srv/home/rjacobs3/anaconda3/envs/testenv2/lib/python3.6/site-packages/detectron2_repo/configs/COCO-InstanceSegmentation/model_weights/faster_rcnn_X_101_32x8d_FPN_3x_model_final_68b088.pkl"
                          }

    if input_yaml['use_pretrained_model_weights'] == True:
        cfg.MODEL.WEIGHTS = model_weights_urls[input_yaml['pretrained_model_weights']]
        # Otherwise will use ImageNet weights instead of model-specific weights on CoCo data

    cfg.DATASETS.TRAIN = (input_yaml['train_dataset_path'],)
    cfg.DATASETS.TEST = (input_yaml['test_dataset_path'],)

    cfg.DATALOADER.NUM_WORKERS = 4  # Default was 2. Try 4?

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = input_yaml['max_iter']

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes: bdot, 111 and 100

    cfg.OUTPUT_DIR = input_yaml['output_dir']

    # HYPERPARAMS TO TUNE
    cfg.MODEL.RPN.IOU_THRESHOLDS = [input_yaml['rpn_iou_min'], input_yaml['rpn_iou_max']]  # Default RPN IoU thresholds. Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [input_yaml['roi_iou_threshold']]

    # Turn on input image augmentation (via cropping and I think image flipping)
    cfg.INPUT.CROP.ENABLED = input_yaml['crop_enabled']
    cfg.INPUT.CROP.SIZE = input_yaml['crop_size']

    # Adjust size of input training images
    #cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(input_yaml['input_min_size_train'])

    # Tune anchor generator sizes, aspect ratios and angles
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [input_yaml['anchor_angles']]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [input_yaml['anchor_sizes']]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [input_yaml['anchor_aspect_ratios']]

    # Number of layers to freeze backbone network
    cfg.MODEL.BACKBONE.FREEZE_AT = input_yaml['num_frozen_layers']

    # Choose between Mask R-CNN (mask is on) or Faster R-CNN (mask is off)
    cfg.MODEL.MASK_ON = input_yaml['mask_on']

    #####
    # TEST 1
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    # cfg.INPUT.MIN_SIZE_TEST = 800
    # cfg.INPUT.MAX_SIZE_TRAIN = 1333
    # cfg.INPUT.MAX_SIZE_TEST = 1333
    # cfg.INPUT.CROP.ENABLED = False
    ######
    # TEST 2 (flipping is disabled in detection_utils.py at bottom)
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    # cfg.INPUT.MIN_SIZE_TEST = 800
    # cfg.INPUT.MAX_SIZE_TRAIN = 1333
    # cfg.INPUT.MAX_SIZE_TEST = 1333
    # cfg.INPUT.CROP.ENABLED = False
    ######

    # cfg.TEST.EVAL_PERIOD = 1000

    # Consider changing for TRAINING:
    # _C.INPUT.CROP.ENABLED = True
    # _C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
    # _C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    # _C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
    # See detectron2/solver/build.py for LR scheduler options
    # _C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    # The period (in terms of steps) to evaluate the model during training.
    # Set to 0 to disable.
    # _C.TEST.EVAL_PERIOD = 0

    # Consider for TESTING:
    # Overlap threshold used for non-maximum suppression (suppress boxes with
    # IoU >= this threshold)
    # _C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    # If True, augment proposals with ground-truth boxes before sampling proposals to
    # train ROI heads.
    # _C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
    # _C.TEST.AUG = CN({"ENABLED": False})
    pprint(cfg)
    return cfg

def make_trainer(cfg, starting_fresh=True):
    if starting_fresh == True:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # only make output dir if starting fresh
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)  # starting fresh
    else:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)  # starting from model saved in cfg.OUTPUT_DIR (default "output")
    return trainer

def run_trainer(trainer):
    trainer.train()
    return

# This is the main function to call to train maskrcnn from detectron2
#def train_detectron_maskrcnn(output_dir, starting_fresh, roi_iou_threshold, rpn_iou_min, rpn_iou_max, anchor_angles,
#                             anchor_sizes, anchor_aspect_ratios, num_frozen_layers, pretrained_model_name,
#                             use_pretrained_model_weights, pretrained_model_weights, crop_enabled, crop_size):
def train_detectron_maskrcnn(input_yaml):
    try:
        defect_metadata = get_defect_metadata(input_yaml['train_dataset_path'],
                                              input_yaml['test_dataset_path'],
                                              input_yaml['train_annotations_path'],
                                              input_yaml['test_annotations_path'])
    except AssertionError:
        print(
            'Defect metadata has already been assigned. If you wish to reset the defect metadata, restart the runtime')
        pass

    #anno_dict_list_train = get_defect_anno_dict_train(train_dataset_path=input_yaml['train_dataset_path'])
    #anno_dict_list_val = get_defect_anno_dict_val(test_dataset_path=input_yaml['test_dataset_path'])

    cfg = get_config_file(input_yaml)
    trainer = make_trainer(cfg=cfg, starting_fresh=input_yaml['starting_fresh'])

    #print('CONFIG FILE')
    #pprint(cfg)
    #print('DEFECT METADATA')
    #pprint(defect_metadata)
    # Run if want to train model
    run_trainer(trainer=trainer)

    return cfg, defect_metadata


