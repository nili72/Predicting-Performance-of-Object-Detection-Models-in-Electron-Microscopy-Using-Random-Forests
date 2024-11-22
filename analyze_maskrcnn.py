from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from pprint import pprint
import sys
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import cv2
import os
import itertools
import json
from copy import copy

from matplotlib.patches import Polygon
from matplotlib.figure import Figure, figaspect
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.font_manager import FontProperties

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import skew

from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

from shapely.geometry import Polygon as shapelyPolygon

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, 'defect_mask_' + str(n))
            # if n < len(list_dfs)-1:
            #  df.to_excel(writer,'defect_mask_'+str(n))
            # else:
            #  df.to_excel(writer, 'image_summary_stats')
        writer.save()
    return

def get_mask_vertices(points):
    #print('POINTS', points)
    try:
        hull = ConvexHull(points)
    except:
        # Usually an error occurs if the input is too low-dimensional due to rounding of pixel values. Joggle input to help avoid this
        hull = ConvexHull(points, qhull_options='QJ')
    vertices = np.array([points[hull.vertices, 0], points[hull.vertices, 1]]).T
    return vertices

def get_fig_ax(aspect_ratio, x_align):
    w, h = figaspect(aspect_ratio)
    fig = Figure(figsize=(w, h))
    FigureCanvas(fig)

    # Set custom positioning, see this guide for more details:
    # https://python4astronomers.github.io/plotting/advanced.html
    left = 0.10
    bottom = 0.15
    right = 0.01
    top = 0.05
    width = x_align - left - right
    height = 1 - bottom - top
    ax = fig.add_axes((left, bottom, width, height), frameon=True)
    fig.set_tight_layout(False)
    return fig, ax

def plot_stats(fig, stats, x_align=0.65, y_align=0.90, font_dict=dict(), fontsize=10, type='float'):
    """
    Method that prints stats onto the plot. Goes off screen if they are too long or too many in number.

    Args:

        fig: (matplotlib figure object), a matplotlib figure object

        stats: (dict), dict of statistics to be included with a plot

        x_align: (float), float denoting x position of where to align display of stats on a plot

        y_align: (float), float denoting y position of where to align display of stats on a plot

        font_dict: (dict), dict of matplotlib font options to alter display of stats on plot

        fontsize: (int), the fontsize of stats to display on plot

    Returns:

        None

    """

    stat_str = '\n'.join(stat_to_string(name, value, nice_names=nice_names(), type=type)
                           for name,value in stats.items())

    fig.text(x_align, y_align, stat_str,
             verticalalignment='top', wrap=True, fontdict=font_dict, fontproperties=FontProperties(size=fontsize))

def stat_to_string(name, value, nice_names, type):
    """
    Method that converts a metric object into a string for displaying on a plot

    Args:

        name: (str), long name of a stat metric or quantity

        value: (float), value of the metric or quantity

    Return:

        (str), a string of the metric name, adjusted to look nicer for inclusion on a plot

    """

    " Stringifies the name value pair for display within a plot "
    if name in nice_names:
        name = nice_names[name]
    else:
        name = name.replace('_', ' ')

    # has a name only
    if not value:
        return name
    # has a mean and std
    if isinstance(value, tuple):
        mean, std = value
        if name == 'MAPE':
            return f'{name}:' + '\n\t' + f'{mean:3.2f}' + r'$\pm$' + f'{std:3.2f}'
        if name == 'R$^2$':
            return f'{name}:' + '\n\t' + f'{mean:3.2f}' + r'$\pm$' + f'{std:3.2f}'
        if name == 'Obs:Pred ratio':
            return f'{name}:' + '\n\t' + f'{mean:3.2f}' + r'$\pm$' + f'{std:3.2f}'
        else:
            return f'{name}:' + '\n\t' + f'{mean:3.2e}' + r'$\pm$' + f'{std:3.2e}'

    # has a name and value only
    if isinstance(value, int) or (isinstance(value, float) and value%1 == 0):
        return f'{name}: {int(value)}'
    if isinstance(value, float):
        if name == 'MAPE':
            return f'{name}: {value:3.2f}'
        if name == 'R$^2$':
            return f'{name}: {value:3.2f}'
        if name == 'Obs:Pred ratio':
            return f'{name}: {value:3.2f}'
        else:
            if type == 'float':
                return f'{name}: {value:3.2f}'
            elif type == 'scientific':
                return f'{name}: {value:3.2e}'
    return f'{name}: {value}' # probably a string

def nice_names():
    nice_names = {
    # classification:
    'accuracy': 'Accuracy',
    'f1_binary': '$F_1$',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_samples': 'f1_samples',
    'f1_weighted': 'f1_weighted',
    'log_loss': 'log_loss',
    'precision_binary': 'Precision',
    'precision_macro': 'prec_macro',
    'precision_micro': 'prec_micro',
    'precision_samples': 'prec_samples',
    'precision_weighted': 'prec_weighted',
    'recall_binary': 'Recall',
    'recall_macro': 'rcl_macro',
    'recall_micro': 'rcl_micro',
    'recall_samples': 'rcl_samples',
    'recall_weighted': 'rcl_weighted',
    'roc_auc': 'ROC_AUC',
    # regression:
    'explained_variance': 'expl_var',
    'mean_absolute_error': 'MAE',
    'mean_squared_error': 'MSE',
    'mean_squared_log_error': 'MSLE',
    'median_absolute_error': 'MedAE',
    'root_mean_squared_error': 'RMSE',
    'rmse_over_stdev': r'RMSE/$\sigma_y$',
    'R2': '$R^2$',
    'R2_noint': '$R^2_{noint}$',
    'R2_adjusted': '$R^2_{adjusted}$',
    'R2_fitted': '$R^2_{fitted}$'
    }
    return nice_names

def str2tuple(string):
    tup = tuple(map(int, string.split(', ')))
    return tup

def get_true_data_stats(cfg, defect_metadata, anno_dict_list_val, filename, model_checkpoint, true_and_pred_matching_threshold,
                        iou_score_threshold_test, show_images=False, save_images=False, save_all_data=False, mask_on=True):
    # Find the right image
    found_image = False
    for i, anno_dict in enumerate(anno_dict_list_val):
        base_filename = anno_dict["file_name"].split('/')[-1]
        if base_filename == filename:
            anno_dict = anno_dict_list_val[i]
            found_image = True
            print('Successfully found image', filename)
            break
    if found_image == False:
        print(
            'WARNING: An error occurred, the provided filename could not be corresponded with the name of an image file in the provided annotation dictionaries')
        return

    img = cv2.imread(anno_dict["file_name"])

    # Assign color to each defect in the image
    assigned_colors_list = list()
    for defect in anno_dict['annotations']:
        id = defect['category_id']
        if id == 0:  # bdot
            assigned_colors_list.append('b')
        elif id == 1:  # 111
            assigned_colors_list.append('r')
        else:
            assigned_colors_list.append('y')
    anno_dict['assigned_colors'] = assigned_colors_list

    if mask_on == True:
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=defect_metadata,
                                scale=0.8,
                                instance_mode=ColorMode.SEGMENTATION
                                )

        vis = visualizer.draw_dataset_dict(anno_dict)
        img2 = vis.get_image()[:, :, ::-1]
        if show_images == True:
            cv2_imshow(img2)
        if save_images == True:
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, filename + '_true_'+'TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_Checkpoint_'+str(model_checkpoint[:-4])+'.png'), img2)

    # PARSE AND OUTPUT TRUE PIXEL DATA
    df_list_true = list()
    true_classes = list()
    true_boxes = list()
    true_segmentations = list()
    n_masks = 0
    true_pixels_all = np.empty((anno_dict['height'], anno_dict['width']))
    true_pixels_all.fill(9999)
    for obj in anno_dict['annotations']:
        data_dict_true = dict()
        seg_y = list()
        seg_x = list()
        seg_y_nearestint = list()
        seg_x_nearestint = list()
        true_classes.append(obj['category_id'])
        true_boxes.append(obj['bbox'])
        #true_segmentations = list()

        n_defects = len(anno_dict['annotations'])
        if mask_on == False:
            for i, box in enumerate(true_boxes):
                # Note that there are no segmentations when mask is off. However, need to populate this list as it is
                # used later. It will carry through but not be used for any analysis.
                true_segmentations.append([[0, 0], [0, 0]])
        if mask_on == True:
            for i, seg in enumerate(obj['segmentation'][0]):
                if i == 0:
                    seg_y.append(seg)
                    seg_y_nearestint.append(int(seg))
                else:
                    if i % 2 == 0:
                        seg_y.append(seg)
                        seg_y_nearestint.append(int(seg))
                    else:
                        seg_x.append(seg)
                        seg_x_nearestint.append(int(seg))

            segmentation = np.array([seg_y, seg_x]).T
            segmentation_nearestint = np.array([seg_y_nearestint, seg_x_nearestint]).T
            data_dict_true['segmentation_y'] = segmentation[:, 0]
            data_dict_true['segmentation_x'] = segmentation[:, 1]
            true_segmentations.append([segmentation_nearestint[:, 0].tolist(), segmentation_nearestint[:, 1].tolist()])

            # Get pixels inside segmentation mask
            vertices = np.array(
                [[obj['bbox'][0], obj['bbox'][1]], [obj['bbox'][0], obj['bbox'][3]], [obj['bbox'][2], obj['bbox'][3]],
                 [obj['bbox'][2], obj['bbox'][1]]])
            # Make the path using the segmentation mask (polygon)
            poly = Polygon(segmentation)
            path = poly.get_path()
            x, y = np.meshgrid(np.arange(min(vertices[:, 0]) - 10, max(vertices[:, 0]) + 10),
                               np.arange(min(vertices[:, 1]) - 10, max(vertices[:, 1]) + 10))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            points_in_mask = path.contains_points(points)

            mask_list = list()
            defectid_list = list()
            pixels_y = list()
            pixels_x = list()
            for i, isin in enumerate(points_in_mask.tolist()):
                if isin == True:
                    pixels_y.append(points[i].tolist()[0])
                    pixels_x.append(points[i].tolist()[1])
            for n in range(len(pixels_y)):
                mask_list.append(n_masks)
                defectid_list.append(obj['category_id'])

            for y, x, defect in zip(pixels_y, pixels_x, defectid_list):
                if y < anno_dict['height']:
                    if x < anno_dict['width']:
                        true_pixels_all[int(y), int(x)] = defect

            # Put the true x, y pixels on the overall image array
            data_dict_true['mask'] = mask_list
            data_dict_true['defect ID'] = defectid_list
            data_dict_true['pixel_list_y'] = pixels_y
            data_dict_true['pixel_list_x'] = pixels_x
            df = pd.DataFrame.from_dict(data_dict_true, orient='index')
            df = df.transpose()
            df_list_true.append(df)
            n_masks += 1

    if save_all_data == True:
        if mask_on == True:
            print('Saving results to excel sheet...')
            save_xls(df_list_true, os.path.join(cfg.OUTPUT_DIR, filename + '_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_true_data_processed.xlsx'))

    print('Have true df list of size', len(true_boxes))

    return true_pixels_all, true_classes, true_segmentations, true_boxes


def get_pred_data_stats(cfg, defect_metadata, anno_dict_list_val, filename, predictor, model_checkpoint,
                        true_and_pred_matching_threshold, iou_score_threshold_test, show_images=False, save_images=False,
                        save_all_data=False, mask_on=True):
    # Find the right image
    found_image = False
    for i, anno_dict in enumerate(anno_dict_list_val):
        base_filename = anno_dict["file_name"].split('/')[-1]
        if base_filename == filename:
            anno_dict = anno_dict_list_val[i]
            found_image = True
            print('Successfully found image', filename)
            break
    if found_image == False:
        print(
            'WARNING: An error occurred, the provided filename could not be corresponded with the name of an image file in the provided annotation dictionaries')
        return

    # Assign color to each defect in the image
    assigned_colors_list = list()
    for defect in anno_dict['annotations']:
        id = defect['category_id']
        if id == 0:  # bdot
            assigned_colors_list.append('b')
        elif id == 1:  # 111
            assigned_colors_list.append('r')
        else:
            assigned_colors_list.append('y')
    anno_dict['assigned_colors'] = assigned_colors_list

    img = cv2.imread(anno_dict["file_name"])
    outputs = predictor(img)

    #TODO: have visualizer also work to plot bounding boxes if mask_on = False.
    #if mask_on == True:
    visualizer = Visualizer(img[:, :, ::-1],
                            metadata=defect_metadata,
                            scale=0.8,
                            instance_mode=ColorMode.SEGMENTATION
                            )
    v = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    img3 = v.get_image()[:, :, ::-1]
    if show_images == True:
        cv2_imshow(img3)
    if save_images == True:
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, filename + '_predicted_'+'TruePredMatch_'+str(true_and_pred_matching_threshold)+
                            '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_Checkpoint_'+str(model_checkpoint[:-4])+'.png'), img3)


    instances = outputs['instances']
    #instances = outputs['instances'].to("cpu")

    #print('INSTANCES')
    #pprint(outputs["instances"].to("cpu"))

    print('ASSIGNED COLORS LENGTH')
    print(len(assigned_colors_list))

    print('OUTPUT INSTANCES')
    pprint(outputs['instances'])

    # PARSE AND OUTPUT PREDICTED PIXEL DATA
    # Organize predictions for each mask (i.e. predicted defect) and save as spreadsheet
    n_masks = 0
    df_list = list()
    pred_classes = list()
    pred_segmentations = list()
    pred_boxes = list()

    pred_pixels_all = np.empty((anno_dict['height'], anno_dict['width']))
    pred_pixels_all.fill(9999)

    if mask_on == False:
        pred_boxes = np.array(instances.pred_boxes).tolist()
        pred_boxes = [box.cpu().numpy().tolist() for box in pred_boxes]
        pred_classes = instances.pred_classes
        pred_classes = [int(c.cpu().numpy()) for c in pred_classes]
        pred_segmentations = [[[0, 0], [0, 0]] for c in pred_classes]

    if mask_on == True:
        # NEW WAY HERE 9/22/20
        #pred_boxes = np.array(instances.pred_boxes).tolist()
        #pred_boxes = [box.cpu().numpy().tolist() for box in pred_boxes]
        #pred_classes = instances.pred_classes
        #pred_classes = [int(c.cpu().numpy()) for c in pred_classes]
        ######

        #masks = np.asarray([mask.to("cpu") for mask in instances.pred_masks])
        #print('MASKS as array')
        #print(masks[0])
        #masks = [GenericMask(x, x.shape[0], x.shape[1]) for x in masks]
        #print('MASKS as GenericMask')
        #print(masks[0])
        #polygons = [m.polygons for m in masks]
        #print('POLYGONS')
        #print(polygons[0])

        for mask in instances.pred_masks:
            try:
                data_dict = dict()
                pred_coords_x = list()
                pred_coords_y = list()
                mask_id = list()
                defect_id = list()
                col_pixel = 0

                # NEW WAY HERE 9/22/20
                #print('MASK')
                #print(mask)
                #mask = np.array(mask.to('cpu'))
                #mask_generic = GenericMask(mask, mask.shape[0], mask.shape[1])
                #print('MASK as GenericMask')
                #print(mask_generic)
                #polygons = mask_generic.polygons
                #print('POLYGONS')
                #print(polygons[0])
                #pred_segmentations.append(polygons[0])
                #sys.exit()
                #############

                pred_boxes.append(np.array(instances.pred_boxes[n_masks].tensor.to("cpu")).tolist()[0])
                pred_classes.append(int(instances.pred_classes[n_masks].to("cpu")))
                #n_masks += 1

                # HERE- a faster way? Yes!
                pred_coords_x = np.where(mask.to("cpu"))[0].tolist()
                pred_coords_y = np.where(mask.to("cpu"))[1].tolist()

                for i in range(len(pred_coords_x)):
                    mask_id.append(n_masks)
                    defect_id.append(int(instances.pred_classes[n_masks].to("cpu")))
                '''
                for col in mask.to("cpu"):
                    row_pixel = 0
                    for row in col:
                        if bool(row) == True:
                            pred_coords_x.append(col_pixel)
                            pred_coords_y.append(row_pixel)
                            mask_id.append(n_masks)
                            defect_id.append(int(instances.pred_classes[n_masks].to("cpu")))
                        row_pixel += 1
                    col_pixel += 1
                '''
                # print('PRED COORDS SIZE VS IMAGE SIZE')
                # print(len(pred_coords_y), len(pred_coords_x), anno_dict['height'], anno_dict['width'])
                for y, x, defect in zip(pred_coords_y, pred_coords_x, defect_id):
                    if y < anno_dict['height']:
                        if x < anno_dict['width']:
                            pred_pixels_all[int(y), int(x)] = defect

                points = np.array([pred_coords_y, pred_coords_x])
                points = points.T
                vertices = get_mask_vertices(points)
                vertices_y = list(vertices[:, 0])
                vertices_x = list(vertices[:, 1])
                vertices_y, vertices_x = (list(t) for t in zip(*sorted(zip(vertices_y, vertices_x))))
                vertices = np.array([vertices_y, vertices_x]).T
                data_dict["y"] = pred_coords_y
                data_dict["x"] = pred_coords_x
                data_dict['segmentation_y'] = vertices[:, 0]
                data_dict['segmentation_x'] = vertices[:, 1]
                pred_segmentations.append([vertices[:, 0].tolist(), vertices[:, 1].tolist()])
                data_dict["mask"] = mask_id
                data_dict["defect ID"] = defect_id
                df = pd.DataFrame.from_dict(data_dict, orient='index')
                df = df.transpose()
                df_list.append(df)
                n_masks += 1
            except:
                print('FOUND ISSUE with IMAGE', filename)
                print('with bbox', np.array(instances.pred_boxes[n_masks].tensor.to("cpu")).tolist()[0])
                print('and object type', int(instances.pred_classes[n_masks].to("cpu")))
                n_masks += 1

    if save_all_data == True:
        if mask_on == True:
            print('Saving results to excel sheet...')
            save_xls(df_list, os.path.join(cfg.OUTPUT_DIR, filename + '_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_predicted_data_processed.xlsx'))

    print('Have pred df list of size', len(pred_boxes))

    return pred_pixels_all, pred_classes, pred_segmentations, pred_boxes

def get_predictor(cfg, model_checkpoint, iou_score_threshold_test, test_dataset_path):
    # Now, we perform inference with the trained model on the defect validation dataset. First, let's create a predictor using the model we just trained:
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_checkpoint)
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = iou_score_threshold_test  # set the testing threshold for this model
    #cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = iou_score_threshold_test #default NMS threshold is 0.5
    cfg.DATASETS.TEST = (test_dataset_path,)
    predictor = DefaultPredictor(cfg)
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    #print('CONFIG FILE')
    #pprint(cfg)
    return predictor

def get_pixel_classification_report(cfg, filename, true_pixels_all, pred_pixels_all):
    # cf = confusion_matrix(true_pixels_all.flatten(), pred_pixels_all.flatten())
    report = classification_report(true_pixels_all.flatten(), pred_pixels_all.flatten(), digits=3,
                                   target_names=['bdot', '111', '100', 'background'])
    report_asdict = classification_report(true_pixels_all.flatten(), pred_pixels_all.flatten(), digits=3,
                                          target_names=['bdot', '111', '100', 'background'], output_dict=True)
    print('PIXEL CLASSIFICATION REPORT')
    print(report)
    report_df = pd.DataFrame(report_asdict)
    report_df.to_excel(os.path.join(cfg.OUTPUT_DIR, filename + '_pixel_classification_report.xlsx'))
    return report_asdict

def get_class_classification_report(cfg, filename, true_classes, pred_classes):
    # cf = confusion_matrix(true_pixels_all.flatten(), pred_pixels_all.flatten())
    report = classification_report(true_classes, pred_classes, digits=3,
                                   target_names=['bdot', '111', '100'])
    report_asdict = classification_report(true_classes, pred_classes, digits=3,
                                          target_names=['bdot', '111', '100'], output_dict=True)
    print('CLASS CLASSIFICATION REPORT')
    print(report)
    report_df = pd.DataFrame(report_asdict)
    report_df.to_excel(os.path.join(cfg.OUTPUT_DIR, filename + '_class_classification_report.xlsx'))
    return report_asdict

def get_defect_size(segmentation, image_name, defect_type):
    nm_per_pixel_70kx = 0.880523
    nm_per_pixel_100kx = 0.869141
    nm_per_pixel_300kx = 0.478516
    nm_per_pixel_500kx = 0.283203
    nm_per_pixel_onzone = 0.1416015625  # 2048x2048

    if "70" in image_name:
        true_distance = nm_per_pixel_70kx
    elif "100" in image_name:
        true_distance = nm_per_pixel_100kx
    elif "300" in image_name:
        true_distance = nm_per_pixel_300kx
    elif "500" in image_name:
        true_distance = nm_per_pixel_500kx
    elif "dalong" in image_name:
        true_distance = nm_per_pixel_500kx
    elif "onzone" in image_name:
        true_distance = nm_per_pixel_onzone
    else:
        print('Could not process image file name for density calculation')
        exit()

    # Using a segmentation mask for a defect (true or pred), calculate the defect radius (in pixels)
    # Get center of the segmentation mask (note the array needs to be 2D)
    #print('SEGMENTATION')
    #print(segmentation)
    segmentation_xy = list()
    seg_x = segmentation[0]
    seg_y = segmentation[1]
    for x, y in zip(seg_x, seg_y):
        segmentation_xy.append([x, y])
    #print('SEG XY')
    #print(segmentation_xy)

    segmentation_xy = np.array(segmentation_xy)
    #print('SEG XY array')
    #print(segmentation_xy.shape)
    #print(segmentation_xy)

    center = np.array([[np.mean(segmentation_xy[:, 0]), np.mean(segmentation_xy[:, 1])]])
    min_distance = min(cdist(segmentation_xy, center))
    min_distance = min_distance[0]
    max_distance = max(cdist(segmentation_xy, center))
    max_distance = max_distance[0]
    # THIS IS OLD METHOD. Note: this method assumes an ellipse, calculates major and minor axes, then averages them.
    #defect_radius_pixels = (min_distance+max_distance)/2
    #defect_radius_nm = defect_radius_pixels*true_distance

    # THIS IS NEW METHOD. Exp method: 111 and 100 loops just use the largest radius (because both are circular, just different orientation). Bdot is r = sqrt(r1*r2)
    if defect_type == 0:
        # This is bdot
        defect_radius_pixels = np.sqrt(min_distance*max_distance)
    elif defect_type == 1:
        # This is 111
        defect_radius_pixels = max_distance
    elif defect_type == 2:
        # This is 100
        defect_radius_pixels = max_distance
    defect_radius_nm = defect_radius_pixels*true_distance
    defect_diameter_nm = 2*defect_radius_nm

    # Need to get defect shape factor. Using Heywood circularity factor, = perimeter / circumference of circle of same area
    #print('ON IMAGE', image_name)
    #print('SEG BEGINNING')
    #print(segmentation_xy[0:3])
    # Need to use convex hull construction of segmentation mask, so points follow an ordered path, making the shape closed
    hull = ConvexHull(segmentation_xy)
    vertices = np.array([segmentation_xy[hull.vertices, 0], segmentation_xy[hull.vertices, 1]]).T
    polygon = shapelyPolygon(vertices)
    perimeter = polygon.length
    area = polygon.area
    radius = np.sqrt(area/np.pi)
    circumference = 2*np.pi*radius
    defect_shape_factor = perimeter/circumference
    #print('PERIMETER, AREA, RADIUS, CIRCUMFERENCE')
    #print(perimeter, area, radius, circumference)
    if area == 0.0:
        print('FOUND AREA ZERO, HERE IS POLYGON and FULL SEG')
        print(polygon)
        print(segmentation_xy)

    return defect_diameter_nm, defect_shape_factor

def get_defect_number_densities(true_classes_all_flattened, pred_classes_all_flattened,
                                num_images, validation_image_filenames, model_checkpoint,
                                iou_score_threshold_test, true_and_pred_matching_threshold, save_path, save_to_file=False):
    # Get true and predicted counts of defect types, and associated defect areal densities across all images

    """
    Image pixel to nm conversion factor

    0501_300kx_1nm_clhaadf3_0010.jpg
    (original image: 1024x1024p; 0.49x0.49 um)
     1 px = 0.478515625 nm

    0501_300kx_1nm_clhaadf3_0014.jpg
    (original image: 1024x1024p; 0.49x0.49 um)
     1 px = 0.478515625 nm

    grid1_roi2_500kx_0p5nm_haadf1_0047.jpg
    (original image: 1024x1024p; 0.29x0.29 um)
     1 px = 0.283203125 nm

    dalong1.jpg
    This is actually named BF X500K, 04.tif in Dalong’s records for future reference. (original image: 1024x1024p; 0.29x0.29 um)
    1 px = 0.283203125 nm

    8ROI_100kx_4100CL_foil1.jpg
    (original image: 1024x1024p; 0.89x0.89 um)
    1 px = 0.869140625 nm

    5401_300kx_1nm_clhaadf3_0020.jpg
    (original image: 1024x1024p; 0.49x0.49 um)
    1 px = 0.478515625 nm

    BF X500K, 06 (2).jpg
    (original image: 2048x2048p; 0.29x0.29 um)
    1 px = 0.1416015625 nm

    grid1_roi1_500kx_0p5nm_haadf1_0025.jpg
    (original image: 1024x1024p; 0.29x0.29 um)
    1 px = 0.283203125 nm

    1ROI_100kx_4100CL_foil1.jpg
    (original image: 1024x1024p; 0.89x0.89 um)
    1 px = 0.869140625 nm

    0501_300kx_1nm_clhaadf3_0028.jpg
    (original image: 1024x1024p; 0.49x0.49 um)
    1 px = 0.478515625 nm

    BF X500K, 09 (3).jpg
    (original image: 2048x2048p; 0.29x0.29 um)
    1 px = 0.283203125 nm

    """
    nm_per_pixel_70kx = 0.880523
    nm_per_pixel_100kx = 0.869141
    nm_per_pixel_300kx = 0.478516
    nm_per_pixel_500kx = 0.283203
    nm_per_pixel_onzone = 0.1416015625  # 2048x2048
    num_70kx = 0
    num_100kx = 0
    num_300kx = 0
    num_500kx = 0
    num_onzone = 0 # This is 2048 pixels
    m_per_nm = 10**-9
    num_done = 0
    for image_name in validation_image_filenames:
        if num_done < num_images:
            print(image_name)
            if "70" in image_name:
                num_70kx += 1
            elif "100" in image_name:
                num_100kx += 1
            elif "300" in image_name:
                num_300kx += 1
            elif "500" in image_name:
                num_500kx += 1
            elif "dalong" in image_name:
                # dalong is same size as 500kx
                num_500kx += 1
            elif "onzone" in image_name:
                num_onzone =+ 1
            else:
                print('Could not process image file name for density calculation')
                exit()
            num_done += 1
    real_area = num_70kx*(nm_per_pixel_70kx*1024)**2 + num_100kx*(nm_per_pixel_100kx*1024)**2 + \
                num_300kx*(nm_per_pixel_300kx*1024)**2 + num_500kx*(nm_per_pixel_500kx*1024)**2 + num_onzone*(nm_per_pixel_onzone*2048)**2

    # Number of images determines total area to average over
    # Fill in correct value once you know it
    #pixels_per_nm = 5
    #real_area = (num_images*1024**2)/(pixels_per_nm**2)
    true_num_bdot = 0
    true_num_111 = 0
    true_num_100 = 0
    pred_num_bdot = 0
    pred_num_111 = 0
    pred_num_100 = 0
    for true in true_classes_all_flattened:
        if true == 0:
            true_num_bdot += 1
        if true == 1:
            true_num_111 += 1
        if true == 2:
            true_num_100 += 1
    for pred in pred_classes_all_flattened:
        if pred == 0:
            pred_num_bdot += 1
        if pred == 1:
            pred_num_111 += 1
        if pred == 2:
            pred_num_100 += 1
    true_density_bdot = true_num_bdot / real_area
    true_density_111 = true_num_111 / real_area
    true_density_100 = true_num_100 / real_area
    pred_density_bdot = pred_num_bdot / real_area
    pred_density_111 = pred_num_111 / real_area
    pred_density_100 = pred_num_100 / real_area

    density_bdot_percenterror = 100 * abs(pred_density_bdot - true_density_bdot) / true_density_bdot
    density_111_percenterror = 100 * abs(pred_density_111 - true_density_111) / true_density_111
    density_100_percenterror = 100 * abs(pred_density_100 - true_density_100) / true_density_100

    average_density_percenterror = np.mean([density_100_percenterror, density_111_percenterror, density_bdot_percenterror])

    datadict = {"true num bdot": true_num_bdot,
                "true num 111": true_num_111,
                "true num 100": true_num_100,
                "true density bdot (#/nm^2)": true_density_bdot,
                "true density 111 (#/nm^2)": true_density_111,
                "true density 100 (#/nm^2)": true_density_100,
                "true density bdot x 10^4 (#/nm^2)": true_density_bdot*10**4,
                "true density 111 x 10^4 (#/nm^2)": true_density_111*10**4,
                "true density 100 x 10^4 (#/nm^2)": true_density_100*10**4,
                "pred num bdot": pred_num_bdot,
                "pred num 111": pred_num_111,
                "pred num 100": pred_num_100,
                "pred density bdot (#/nm^2)": pred_density_bdot,
                "pred density 111 (#/nm^2)": pred_density_111,
                "pred density 100 (#/nm^2)": pred_density_100,
                "pred density bdot x 10^4 (#/nm^2)": pred_density_bdot*10**4,
                "pred density 111 x 10^4 (#/nm^2)": pred_density_111*10**4,
                "pred density 100 x 10^4 (#/nm^2)": pred_density_100*10**4,
                "percent error density bdot": density_bdot_percenterror,
                "percent error density 111": density_111_percenterror,
                "percent error density 100": density_100_percenterror,
                "average density percent error": average_density_percenterror,
                "model checkpoint": model_checkpoint,
                "iou_score_threshold_test": iou_score_threshold_test,
                "true_and_pred_matching_threshold": true_and_pred_matching_threshold}
    # Save datadict to excel
    df_defectnumbers = pd.DataFrame().from_dict(datadict, orient='index')
    #print('DEFECT NUMBERS AND DENSITIES')
    #pprint(datadict)
    if save_to_file == True:
        df_defectnumbers.to_excel(save_path + '.xlsx')

    return df_defectnumbers

def get_overall_defect_stats(num_true_perimage, num_pred_perimage, num_found_perimage, model_checkpoint,
                                iou_score_threshold_test, true_and_pred_matching_threshold, save_path, save_to_file=False):
    # Total up the number of instances that are true, predicted, and found correctly for overall P, R, F1 scores
    num_true_total = np.sum(num_true_perimage)
    num_pred_total = np.sum(num_pred_perimage)
    num_found_total = np.sum(num_found_perimage)
    overall_fp = num_pred_total - num_found_total
    overall_fn = num_true_total - num_found_total
    overall_prec = num_found_total / (num_found_total + overall_fp)
    overall_recall = num_found_total / (num_found_total + overall_fn)
    overall_f1 = (2 * overall_prec * overall_recall) / (overall_prec + overall_recall)
    overall_stats_arr = np.array(
        [[num_true_total, num_pred_total, num_found_total, overall_prec, overall_recall, overall_f1, model_checkpoint, iou_score_threshold_test, true_and_pred_matching_threshold]])

    df_overallstats = pd.DataFrame(data=overall_stats_arr,
                                   columns=['num true total', 'num pred total', 'num found total',
                                            'overall precision', 'overall recall', 'overall F1', 'model_checkpoint',
                                            'iou_score_threshold_test', 'true_and_pred_matching_threshold'],
                                   index=['overall stats'])
    if save_to_file == True:
        df_overallstats.to_excel(save_path + '.xlsx')
    return df_overallstats

def get_confusionmatrix_defectID(t_111_p_111, t_111_p_bdot, t_111_p_100,
                                 t_bdot_p_111, t_bdot_p_bdot, t_bdot_p_100,
                                t_100_p_111, t_100_p_bdot, t_100_p_100,
                                model_checkpoint, iou_score_threshold_test, true_and_pred_matching_threshold,
                                save_path, save_to_file=False):
    # Construct confusion matrix of identified defect types
    # Old arrangement
    # cm_arr = np.array([[t_bdot_p_bdot, t_111_p_bdot, t_100_p_bdot],
    #                   [t_bdot_p_111, t_111_p_111, t_100_p_111],
    #                   [t_bdot_p_100, t_111_p_100, t_100_p_100]])
    # New arrangement
    cm_arr = np.array([[t_111_p_111, t_bdot_p_111, t_100_p_111],
                       [t_111_p_bdot, t_bdot_p_bdot, t_100_p_bdot],
                       [t_111_p_100, t_bdot_p_100, t_100_p_100],
                       [model_checkpoint, iou_score_threshold_test, true_and_pred_matching_threshold]])
    # Old names
    # df = pd.DataFrame(data=cm_arr, columns=['True bdot', 'True 111', 'True 100'],
    #                  index=['Pred bdot', 'Pred 111', 'Pred 100'])
    # New names
    df_confusionmatrixdefectID = pd.DataFrame(data=cm_arr, columns=['True 111', 'True bdot', 'True 100'],
                                              index=['Pred 111', 'Pred bdot', 'Pred 100', 'model_checkpoint;iou_score_threshold_test;true_and_pred_matching_threshold'])

    if save_to_file == True:
        df_confusionmatrixdefectID.to_excel(save_path + '.xlsx')

    return df_confusionmatrixdefectID

def get_computedstats_defectID(t_111_p_111, t_111_p_bdot, t_111_p_100,
                                 t_bdot_p_111, t_bdot_p_bdot, t_bdot_p_100,
                                t_100_p_111, t_100_p_bdot, t_100_p_100, model_checkpoint,
                                iou_score_threshold_test, true_and_pred_matching_threshold,
                              save_path, save_to_file=False):
    # Get prec/recall/F1 of identified defect types
    try:
        recall_bdot = t_bdot_p_bdot / (t_bdot_p_bdot + t_bdot_p_111 + t_bdot_p_100)
    except ZeroDivisionError:
        recall_bdot = float('NaN')
    try:
        recall_111 = t_111_p_111 / (t_111_p_111 + t_111_p_bdot + t_111_p_100)
    except ZeroDivisionError:
        recall_111 = float('NaN')
    try:
        recall_100 = t_100_p_100 / (t_100_p_100 + t_100_p_bdot + t_100_p_111)
    except ZeroDivisionError:
        recall_100 = float('NaN')
    try:
        prec_bdot = t_bdot_p_bdot / (t_100_p_bdot + t_bdot_p_bdot + t_111_p_bdot)
    except ZeroDivisionError:
        prec_bdot = float('NaN')
    try:
        prec_111 = t_111_p_111 / (t_111_p_111 + t_100_p_111 + t_bdot_p_111)
    except ZeroDivisionError:
        prec_111 = float('NaN')
    try:
        prec_100 = t_100_p_100 / (t_111_p_100 + t_100_p_100 + t_bdot_p_100)
    except ZeroDivisionError:
        prec_100 = float('NaN')
    try:
        f1_bdot = (2 * prec_bdot * recall_bdot) / (prec_bdot + recall_bdot)
    except ZeroDivisionError:
        f1_bdot = float('NaN')
    try:
        f1_111 = (2 * prec_111 * recall_111) / (prec_111 + recall_111)
    except ZeroDivisionError:
        f1_111 = float('NaN')
    try:
        f1_100 = (2 * prec_100 * recall_100) / (prec_100 + recall_100)
    except ZeroDivisionError:
        f1_100 = float('NaN')
    try:
        recall_overall = (recall_bdot + recall_100 + recall_111) / 3
    except:
        recall_overall = float('NaN')
    try:
        prec_overall = (prec_bdot + prec_100 + prec_111) / 3
    except:
        prec_overall = float('NaN')
    try:
        f1_overall = (f1_bdot + f1_111 + f1_100) / 3
    except:
        f1_overall = float('NaN')

    stats_arr = np.array([[prec_bdot, prec_111, prec_100, prec_overall],
                          [recall_bdot, recall_111, recall_100, recall_overall],
                          [f1_bdot, f1_111, f1_100, f1_overall],
                          [model_checkpoint, iou_score_threshold_test, true_and_pred_matching_threshold, 'n/a']])

    df_computedstatsdefectID = pd.DataFrame(data=stats_arr, columns=['bdot', '111', '100', 'overall'],
                                            index=['precision', 'recall', 'F1 score', 'model_checkpoint;iou_score_threshold_test;true_and_pred_matching_threshold'])
    if save_to_file == True:
        df_computedstatsdefectID.to_excel(save_path + '.xlsx')

    return df_computedstatsdefectID

def get_defect_sizes_average_and_errors(true_defectsizes_bdot_nm, true_defectsizes_111_nm, true_defectsizes_100_nm,
                                        pred_defectsizes_bdot_nm, pred_defectsizes_111_nm, pred_defectsizes_100_nm,
                                        true_defectshapes_bdot, true_defectshapes_111, true_defectshapes_100,
                                        pred_defectshapes_bdot, pred_defectshapes_111, pred_defectshapes_100,
                                        model_checkpoint, iou_score_threshold_test, true_and_pred_matching_threshold,
                                        save_path, save_to_file, cfg, file_string):
    # Get average defect radius per defect, output to excel file
    average_true_defectsizes_bdot_nm = np.mean(true_defectsizes_bdot_nm)
    average_true_defectsizes_111_nm = np.mean(true_defectsizes_111_nm)
    average_true_defectsizes_100_nm = np.mean(true_defectsizes_100_nm)
    average_pred_defectsizes_bdot_nm = np.mean(pred_defectsizes_bdot_nm)
    average_pred_defectsizes_111_nm = np.mean(pred_defectsizes_111_nm)
    average_pred_defectsizes_100_nm = np.mean(pred_defectsizes_100_nm)

    average_true_defectshapes_bdot = np.mean(true_defectshapes_bdot)
    average_true_defectshapes_111 = np.mean(true_defectshapes_111)
    average_true_defectshapes_100 = np.mean(true_defectshapes_100)
    average_pred_defectshapes_bdot = np.mean(pred_defectshapes_bdot)
    average_pred_defectshapes_111 = np.mean(pred_defectshapes_111)
    average_pred_defectshapes_100 = np.mean(pred_defectshapes_100)

    percent_error_defectsizes_bdot = 100 * abs(
        average_true_defectsizes_bdot_nm - average_pred_defectsizes_bdot_nm) / average_true_defectsizes_bdot_nm
    percent_error_defectsizes_111 = 100 * abs(
        average_true_defectsizes_111_nm - average_pred_defectsizes_111_nm) / average_true_defectsizes_111_nm
    percent_error_defectsizes_100 = 100 * abs(
        average_true_defectsizes_100_nm - average_pred_defectsizes_100_nm) / average_true_defectsizes_100_nm

    percent_error_defectshapes_bdot = 100 * abs(
        average_true_defectshapes_bdot - average_pred_defectshapes_bdot) / average_true_defectshapes_bdot
    percent_error_defectshapes_111 = 100 * abs(
        average_true_defectshapes_111 - average_pred_defectshapes_111) / average_true_defectshapes_111
    percent_error_defectshapes_100 = 100 * abs(
        average_true_defectshapes_100 - average_pred_defectshapes_100) / average_true_defectshapes_100

    average_percent_error_defect_sizes = np.mean([percent_error_defectsizes_bdot, percent_error_defectsizes_111, percent_error_defectsizes_100])
    average_percent_error_defect_shapes = np.mean([percent_error_defectshapes_bdot, percent_error_defectshapes_111, percent_error_defectshapes_100])

    sizes_arr = np.array([[true_defectsizes_bdot_nm, true_defectsizes_111_nm, true_defectsizes_100_nm],
                          [len(true_defectsizes_bdot_nm), len(true_defectsizes_111_nm), len(true_defectsizes_100_nm)],
                          [average_true_defectsizes_bdot_nm, average_true_defectsizes_111_nm,
                           average_true_defectsizes_100_nm],
                          [pred_defectsizes_bdot_nm, pred_defectsizes_111_nm, pred_defectsizes_100_nm],
                          [len(pred_defectsizes_bdot_nm), len(pred_defectsizes_111_nm), len(pred_defectsizes_100_nm)],
                          [average_pred_defectsizes_bdot_nm, average_pred_defectsizes_111_nm,
                           average_pred_defectsizes_100_nm],
                          [percent_error_defectsizes_bdot, percent_error_defectsizes_111,
                           percent_error_defectsizes_100],
                          [average_percent_error_defect_sizes, 'n/a', 'n/a'],
                          [percent_error_defectshapes_bdot, percent_error_defectshapes_111,
                           percent_error_defectshapes_100],
                          [average_percent_error_defect_shapes, 'n/a', 'n/a'],
                          [model_checkpoint, 'n/a', 'n/a'],
                          [iou_score_threshold_test, 'n/a', 'n/a'],
                          [true_and_pred_matching_threshold, 'n/a', 'n/a']])

    df_defectsizes = pd.DataFrame(data=sizes_arr, columns=['bdot', '111', '100'],
                                  index=['true all diameter values',
                                         'number true diameter values',
                                         'true averaged diameter values',
                                         'predicted all diameter values',
                                         'number predicted diameter values',
                                         'predicted average diameter values',
                                         'percent error average diameter values',
                                         'average percent error all defect diameters',
                                         'percent error average shape values',
                                         'average percent error all defect shapes',
                                         'model_checkpoint',
                                         'iou_score_threshold_test',
                                         'true_and_pred_matching_threshold'])
    if save_to_file == True:
        df_defectsizes.to_excel(save_path + '.xlsx')

    #####
    #
    # Here, make histograms of true and predicted defect size distributions
    #
    #####
    # Black dot size histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectsizes_bdot_nm, bins=np.arange(0, 20, 2), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectsizes_bdot_nm, bins=np.arange(0, 20, 2), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Black dot sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectsizes_bdot_nm), range(len(true_defectsizes_bdot_nm)), 'b--', linewidth=1, label='True')
    ax2.set_ylabel('Total number of true black dots', fontsize=12)
    ax2.plot(sorted(pred_defectsizes_bdot_nm), range(len(pred_defectsizes_bdot_nm)), 'g--', linewidth=1, label='Predicted')
    ax2.set_ylabel('Total number of predicted black dots', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(true_defectsizes_bdot_nm)
    pred_skew = skew(pred_defectsizes_bdot_nm)
    true_stats = pd.DataFrame(true_defectsizes_bdot_nm).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectsizes_bdot_nm).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_Blackdot_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    # 111 size histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectsizes_111_nm, bins=np.arange(0, 100, 10), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectsizes_111_nm, bins=np.arange(0, 100, 10), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<111> loop sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectsizes_111_nm), range(len(true_defectsizes_111_nm)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true <111> loops', fontsize=12)
    ax2.plot(sorted(pred_defectsizes_111_nm), range(len(pred_defectsizes_111_nm)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted <111> loops', fontsize=12)
    ax.legend(loc='best')
    true_skew = skew(true_defectsizes_111_nm)
    pred_skew = skew(pred_defectsizes_111_nm)
    true_stats = pd.DataFrame(true_defectsizes_111_nm).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectsizes_111_nm).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_111loops_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    # 100 size histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectsizes_100_nm, bins=np.arange(0, 100, 10), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectsizes_100_nm, bins=np.arange(0, 100, 10), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<100> loop sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectsizes_100_nm), range(len(true_defectsizes_100_nm)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true <100> loops', fontsize=12)
    ax2.plot(sorted(pred_defectsizes_100_nm), range(len(pred_defectsizes_100_nm)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted <100> loops', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(true_defectsizes_100_nm)
    pred_skew = skew(pred_defectsizes_100_nm)
    true_stats = pd.DataFrame(true_defectsizes_100_nm).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectsizes_100_nm).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_100loops_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    # All defect sizes together histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    all_true_defects = list(true_defectsizes_100_nm) + list(true_defectsizes_111_nm) + list(true_defectsizes_bdot_nm)
    all_pred_defects = list(pred_defectsizes_100_nm) + list(pred_defectsizes_111_nm) + list(pred_defectsizes_bdot_nm)
    ax.hist(all_true_defects, bins=np.arange(0, 100, 10), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(all_pred_defects, bins=np.arange(0, 100, 10), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Defect sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(all_true_defects), range(len(all_true_defects)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true defects', fontsize=12)
    ax2.plot(sorted(all_pred_defects), range(len(all_pred_defects)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted defects', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(all_true_defects)
    pred_skew = skew(all_pred_defects)
    true_stats = pd.DataFrame(all_true_defects).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(all_pred_defects).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_AllDefects_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    #####
    #
    # Here, make histograms of true and predicted defect shape distributions
    #
    #####
    # Black dot shape histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectshapes_bdot, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectshapes_bdot, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Black dot Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectshapes_bdot), range(len(true_defectshapes_bdot)), 'b--', linewidth=1, label='True')
    ax2.set_ylabel('Total number of true black dots', fontsize=12)
    ax2.plot(sorted(pred_defectshapes_bdot), range(len(pred_defectshapes_bdot)), 'g--', linewidth=1, label='Predicted')
    ax2.set_ylabel('Total number of predicted black dots', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(true_defectshapes_bdot)
    pred_skew = skew(pred_defectshapes_bdot)
    true_stats = pd.DataFrame(true_defectshapes_bdot).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectshapes_bdot).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    ax.set_yscale('log')
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_Blackdot_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    # 111 shape histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectshapes_111, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectshapes_111, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<111> loop Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectshapes_111), range(len(true_defectshapes_111)), 'b--', linewidth=1, label='True')
    ax2.set_ylabel('Total number of true <111> loops', fontsize=12)
    ax2.plot(sorted(pred_defectshapes_111), range(len(pred_defectshapes_111)), 'g--', linewidth=1, label='Predicted')
    ax2.set_ylabel('Total number of predicted <111> loops', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(true_defectshapes_111)
    pred_skew = skew(pred_defectshapes_111)
    true_stats = pd.DataFrame(true_defectshapes_111).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectshapes_111).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    ax.set_yscale('log')
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_111Loops_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    # 100 shape histogram
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_defectshapes_100, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_defectshapes_100, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<100> loop Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_defectshapes_100), range(len(true_defectshapes_100)), 'b--', linewidth=1, label='True')
    ax2.set_ylabel('Total number of true <100> loops', fontsize=12)
    ax2.plot(sorted(pred_defectshapes_100), range(len(pred_defectshapes_100)), 'g--', linewidth=1, label='Predicted')
    ax2.set_ylabel('Total number of predicted <100> loops', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(true_defectshapes_100)
    pred_skew = skew(pred_defectshapes_100)
    true_stats = pd.DataFrame(true_defectshapes_100).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(pred_defectshapes_100).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    ax.set_yscale('log')
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_100Loops_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    # All shapes histogram
    all_true_shapes = true_defectshapes_100 + true_defectshapes_111 + true_defectshapes_bdot
    all_pred_shapes = pred_defectshapes_100 + pred_defectshapes_111 + pred_defectshapes_bdot
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(all_true_shapes, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(all_pred_shapes, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Defect Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(all_true_shapes), range(len(all_true_shapes)), 'b--', linewidth=1, label='True')
    ax2.set_ylabel('Total number of true defects', fontsize=12)
    ax2.plot(sorted(all_pred_shapes), range(len(all_pred_shapes)), 'g--', linewidth=1, label='Predicted')
    ax2.set_ylabel('Total number of predicted defects', fontsize=12)
    ax.legend(loc='lower right')
    true_skew = skew(all_true_shapes)
    pred_skew = skew(all_pred_shapes)
    true_stats = pd.DataFrame(all_true_shapes).describe().to_dict()[0]
    true_stats['skew'] = true_skew
    pred_stats = pd.DataFrame(all_pred_shapes).describe().to_dict()[0]
    pred_stats['skew'] = pred_skew
    plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color':'b'})
    plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color':'g'})
    ax.set_yscale('log')
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'AllImages_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_AllDefects_'+str(file_string)+'.png'),
                dpi=250, bbox_inches='tight')

    return df_defectsizes

def match_true_and_predicted_defects_pixelnorm_mask(true_segmentations_oneimage_abbrev, pred_segmentations_oneimage_abbrev,
        true_classes_all_oneimage_sorted, pred_classes_all_oneimage_sorted,
        true_segmentations_oneimage_sorted, pred_segmentations_oneimage_sorted,
        num_found, t_111_p_111, t_111_p_bdot, t_111_p_100,
        t_bdot_p_111, t_bdot_p_bdot, t_bdot_p_100,
        t_100_p_111, t_100_p_bdot, t_100_p_100,
        true_defectsizes_bdot_nm_foundonly, true_defectsizes_111_nm_foundonly, true_defectsizes_100_nm_foundonly,
        pred_defectsizes_bdot_nm_foundonly, pred_defectsizes_111_nm_foundonly, pred_defectsizes_100_nm_foundonly,
        image_name, pixelnorm_threshold=60):

    # Loop over true segs and check if they correspond to pred segs. If not, then prediction missed one
    true_pred_index_list = list()
    for i, true_seg in enumerate(true_segmentations_oneimage_abbrev):
        # print('ANALYZING TRUE SEG NUMBER', i)
        found_y = False
        found_x = False
        norms = dict()
        for j, pred_seg in enumerate(pred_segmentations_oneimage_abbrev):
            if j not in true_pred_index_list:
                # print('Consider pred seg number', j)
                if abs(true_seg[0] - pred_seg[0]) <= pixelnorm_threshold:
                    found_y = True
                    if abs(true_seg[1] - pred_seg[1]) <= pixelnorm_threshold:
                        found_x = True
                        norm = np.sqrt((abs(true_seg[0] - pred_seg[0])) ** 2 + (abs(true_seg[1] - pred_seg[1])) ** 2)
                        # print('norm is now', norm)
                        norms[j] = norm
        # Use whichever has smallest norm
        norm = 10000
        for k, v in norms.items():
            if v < norm:
                norm = v
                true_pred_index = k

        if found_y == True:
            # print(found_y, found_x)
            if found_x == True:
                true_class = true_classes_all_oneimage_sorted[i]
                pred_class = pred_classes_all_oneimage_sorted[true_pred_index]

                # Found a defect, so overall a true positive (not discerning defect type)
                num_found += 1

                # Recall class labels are 0:bdot, 1:111, 2:100
                # Populate values for defect ID confusion matrix
                if true_class == 0:
                    if pred_class == 0:
                        t_bdot_p_bdot += 1
                    elif pred_class == 1:
                        t_bdot_p_111 += 1
                    elif pred_class == 2:
                        t_bdot_p_100 += 1
                elif true_class == 1:
                    if pred_class == 0:
                        t_111_p_bdot += 1
                    elif pred_class == 1:
                        t_111_p_111 += 1
                    elif pred_class == 2:
                        t_111_p_100 += 1
                elif true_class == 2:
                    if pred_class == 0:
                        t_100_p_bdot += 1
                    elif pred_class == 1:
                        t_100_p_111 += 1
                    elif pred_class == 2:
                        t_100_p_100 += 1

                true_pred_index_list.append(true_pred_index)

                # Calculate the defect size since found a defect where there should be one
                true_defect_diameter, true_defect_shape_factor = get_defect_size(segmentation=true_segmentations_oneimage_sorted[i],
                                                       image_name=image_name, defect_type=true_class)
                pred_defect_diameter, pred_defect_shape_factor = get_defect_size(segmentation=pred_segmentations_oneimage_sorted[true_pred_index],
                                                       image_name=image_name, defect_type=pred_class)

                if true_class == 0:
                    true_defectsizes_bdot_nm.append(true_defect_diameter)
                elif true_class == 1:
                    true_defectsizes_111_nm.append(true_defect_diameter)
                elif true_class == 2:
                    true_defectsizes_100_nm.append(true_defect_diameter)

                if pred_class == 0:
                    pred_defectsizes_bdot_nm.append(pred_defect_diameter)
                elif pred_class == 1:
                    pred_defectsizes_111_nm.append(pred_defect_diameter)
                elif pred_class == 2:
                    pred_defectsizes_100_nm.append(pred_defect_diameter)

                print('FOUND TRUE MASK', [true_seg[0], true_seg[1]], 'with TRUE CLASS',
                      true_class, 'and PRED MASK',
                      [pred_segmentations_oneimage_abbrev[true_pred_index][0],
                       pred_segmentations_oneimage_abbrev[true_pred_index][1]], 'with PRED CLASS',
                      pred_class)
            else:
                print('FOUND TRUE MASK', [true_seg[0], true_seg[1]], 'with TRUE CLASS',
                      true_classes_all_oneimage_sorted[i], 'BUT NO CORRESPONDING PRED MASK')

                # pred_classes_all_oneimage_sorted.insert(i, 3)
        else:
            print('FOUND TRUE MASK', [true_seg[0], true_seg[1]], 'with TRUE CLASS',
                  true_classes_all_oneimage_sorted[i], 'BUT NO CORRESPONDING PRED MASK')

            # pred_classes_all_oneimage_sorted.insert(i, 3)
    return num_found, t_111_p_111, t_111_p_bdot, t_111_p_100, t_bdot_p_111, t_bdot_p_bdot, t_bdot_p_100, \
            t_100_p_111, t_100_p_bdot, t_100_p_100, true_defectsizes_bdot_nm, true_defectsizes_111_nm, \
            true_defectsizes_100_nm, pred_defectsizes_bdot_nm, pred_defectsizes_111_nm, pred_defectsizes_100_nm

def match_true_and_predicted_defects_iou_bbox(true_classes_all_oneimage_sorted, pred_classes_all_oneimage_sorted,
        true_segmentations_oneimage_sorted, pred_segmentations_oneimage_sorted,
        true_boxes_oneimage_sorted, pred_boxes_oneimage_sorted,
        num_found, true_defectsizes_bdot_nm_foundonly, true_defectsizes_111_nm_foundonly, true_defectsizes_100_nm_foundonly,
        pred_defectsizes_bdot_nm_foundonly, pred_defectsizes_111_nm_foundonly, pred_defectsizes_100_nm_foundonly,
        true_defectshapes_bdot_foundonly, true_defectshapes_111_foundonly, true_defectshapes_100_foundonly,
        pred_defectshapes_bdot_foundonly, pred_defectshapes_111_foundonly, pred_defectshapes_100_foundonly,
        image_name, mask_on=True, iou_threshold=0.5):

    t_111_p_111 = 0
    t_111_p_bdot = 0
    t_111_p_100 = 0
    t_bdot_p_111 = 0
    t_bdot_p_bdot = 0
    t_bdot_p_100 = 0
    t_100_p_111 = 0
    t_100_p_bdot = 0
    t_100_p_100 = 0

    # Loop over true bboxes and check if they correspond to pred bboxes. Do this by calculating IoU of all predicted boxes
    # and selecting the highest one. If not, then prediction missed one
    true_pred_index_list = list()
    for i, true_box in enumerate(true_boxes_oneimage_sorted):
        ious = dict()
        for j, pred_box in enumerate(pred_boxes_oneimage_sorted):
            if j not in true_pred_index_list:
                iou = bb_intersection_over_union(boxA=true_box, boxB=pred_box)
                #print('True box', true_box, 'and pred box', pred_box, 'have iou', iou)
                ious[j] = iou
        # Use whichever has largest iou
        iou = -1
        for k, v in ious.items():
            if v > iou:
                iou = v
                true_pred_index = k

        # Check that the iou satisfies the iou_threshold value set by user
        if iou >= iou_threshold:
            true_class = true_classes_all_oneimage_sorted[i]
            pred_class = pred_classes_all_oneimage_sorted[true_pred_index]

            # Found a defect, so overall a true positive (not discerning defect type)
            num_found += 1

            # Recall class labels are 0:bdot, 1:111, 2:100
            # Populate values for defect ID confusion matrix
            if true_class == 0:
                if pred_class == 0:
                    t_bdot_p_bdot += 1
                elif pred_class == 1:
                    t_bdot_p_111 += 1
                elif pred_class == 2:
                    t_bdot_p_100 += 1
            elif true_class == 1:
                if pred_class == 0:
                    t_111_p_bdot += 1
                elif pred_class == 1:
                    t_111_p_111 += 1
                elif pred_class == 2:
                    t_111_p_100 += 1
            elif true_class == 2:
                if pred_class == 0:
                    t_100_p_bdot += 1
                elif pred_class == 1:
                    t_100_p_111 += 1
                elif pred_class == 2:
                    t_100_p_100 += 1

            true_pred_index_list.append(true_pred_index)

            # Calculate the defect size since found a defect where there should be one
            if mask_on == True:
                true_defect_diameter, true_defect_shape_factor = get_defect_size(segmentation=true_segmentations_oneimage_sorted[i],
                                                       image_name=image_name, defect_type=true_class)
                pred_defect_diameter, pred_defect_shape_factor = get_defect_size(segmentation=pred_segmentations_oneimage_sorted[true_pred_index],
                                                       image_name=image_name, defect_type=pred_class)
            else:
                # Note that the use of the mask is required to get defect sizes
                true_defect_diameter = 0
                pred_defect_diameter = 0
                true_defect_shape_factor = 0
                pred_defect_shape_factor = 0

            if true_class == 0:
                true_defectsizes_bdot_nm_foundonly.append(true_defect_diameter)
                true_defectshapes_bdot_foundonly.append(true_defect_shape_factor)
            elif true_class == 1:
                true_defectsizes_111_nm_foundonly.append(true_defect_diameter)
                true_defectshapes_111_foundonly.append(true_defect_shape_factor)
            elif true_class == 2:
                true_defectsizes_100_nm_foundonly.append(true_defect_diameter)
                true_defectshapes_100_foundonly.append(true_defect_shape_factor)

            if pred_class == 0:
                pred_defectsizes_bdot_nm_foundonly.append(pred_defect_diameter)
                pred_defectshapes_bdot_foundonly.append(pred_defect_shape_factor)
            elif pred_class == 1:
                pred_defectsizes_111_nm_foundonly.append(pred_defect_diameter)
                pred_defectshapes_111_foundonly.append(pred_defect_shape_factor)
            elif pred_class == 2:
                pred_defectsizes_100_nm_foundonly.append(pred_defect_diameter)
                pred_defectshapes_100_foundonly.append(pred_defect_shape_factor)

            print('FOUND TRUE BOX', [true_box[0], true_box[1], true_box[2], true_box[3]], 'with TRUE CLASS',
                  true_class, 'and PRED BOX',
                  [pred_boxes_oneimage_sorted[true_pred_index][0], pred_boxes_oneimage_sorted[true_pred_index][1],
                   pred_boxes_oneimage_sorted[true_pred_index][2], pred_boxes_oneimage_sorted[true_pred_index][3]], 'with PRED CLASS',
                  pred_class)
        else:
            print('FOUND TRUE BOX', [true_box[0], true_box[1], true_box[2], true_box[3]], 'with TRUE CLASS',
                  true_classes_all_oneimage_sorted[i], 'BUT NO CORRESPONDING PRED MASK THAT MET IOU THRESHOLD')

    #else:
    #    print('FOUND TRUE MASK', [true_box[0], true_box[1], true_box[2], true_box[3]], 'with TRUE CLASS',
    #          true_classes_all_oneimage_sorted[i], 'BUT NO CORRESPONDING PRED MASK')

    return num_found, t_111_p_111, t_111_p_bdot, t_111_p_100, t_bdot_p_111, t_bdot_p_bdot, t_bdot_p_100, \
            t_100_p_111, t_100_p_bdot, t_100_p_100, true_defectsizes_bdot_nm_foundonly, true_defectsizes_111_nm_foundonly, \
            true_defectsizes_100_nm_foundonly, pred_defectsizes_bdot_nm_foundonly, pred_defectsizes_111_nm_foundonly, pred_defectsizes_100_nm_foundonly, \
            true_defectshapes_bdot_foundonly, true_defectshapes_111_foundonly, true_defectshapes_100_foundonly, \
            pred_defectshapes_bdot_foundonly, pred_defectshapes_111_foundonly, pred_defectshapes_100_foundonly

def bb_intersection_over_union(boxA, boxB):
    # This code is not mine. I got it from this github post: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc,
    # which corrected an issue in this original post: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0.0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def save_excel_together_finalreport_BROKEN(full_dict_dfs_per_IoUscorethreshold, sheet_names, save_path):
    """
    overall_stats_per_IoU (dict): keys are iou_score_threshold_test, values are lists of df's containing all analysis
    """
    writer = pd.ExcelWriter(save_path, engine="openpxyl", mode='a')
    #with ExcelWriter(save_path) as writer:
    num_iou_score_thresholds_done = 0
    for iou_score_threshold_test, full_dict_dfs in full_dict_dfs_per_IoUscorethreshold.items():
        print('on iou score thresh', iou_score_threshold_test)
        num_checkpoints_done = 0
        for model_checkpoint, dict_dfs in full_dict_dfs.items():
            print('on checkpoint', model_checkpoint)
            num_match_true_pred_thresh_done = 0
            for match_true_pred_thresh, list_dfs in dict_dfs.items():
                print('on match true/pred', match_true_pred_thresh)
                for sheet_name, df in zip(sheet_names, list_dfs):
                    print('on sheet name', sheet_name)
                    # omg help me obi-wan kenobi, you're my only hope
                    sheet = writer.sheets[sheet_name]
                    startrow_multiplier = df.shape[0]*(num_match_true_pred_thresh_done+num_checkpoints_done+num_iou_score_thresholds_done)+num_match_true_pred_thresh_done+num_checkpoints_done+num_iou_score_thresholds_done
                    #startcol_multiplier = df.shape[1]*(1+num_match_true_pred_thresh_done+num_checkpoints_done+num_iou_score_thresholds_done)
                    if num_iou_score_thresholds_done == 0:
                        sheet.write(0, 0, "IoU predictor test threshold = " + str(iou_score_threshold_test))
                    else:
                        sheet.write(startrow_multiplier, 0, "IoU predictor test threshold = " + str(iou_score_threshold_test))
                    if num_checkpoints_done == 0:
                        sheet.write(1, 1, "Model checkpoint = "+str(model_checkpoint))
                    else:
                        sheet.write(startrow_multiplier, 1, "Model checkpoint = " + str(model_checkpoint))
                    if num_match_true_pred_thresh_done == 0:
                        sheet.write(2, 2, "Match True and Predicted Threshold = " + str(match_true_pred_thresh))
                    else:
                        sheet.write(startrow_multiplier, 2, "Match True and Predicted Threshold = " + str(match_true_pred_thresh))
                    if (num_iou_score_thresholds_done == 0 and num_checkpoints_done == 0 and num_match_true_pred_thresh_done == 0):
                        df.to_excel(writer, sheet_name, startrow=3, startcol=3)
                    else:
                        df.to_excel(writer, sheet_name, startrow=startrow_multiplier, startcol=3)
                num_match_true_pred_thresh_done += 1
            num_checkpoints_done += 1
        num_iou_score_thresholds_done += 1
    writer.save()
    return

def save_excel_together_finalreport(full_dict_dfs_per_IoUscorethreshold, sheet_names, save_path):
    """
    overall_stats_per_IoU (dict): keys are iou_score_threshold_test, values are lists of df's containing all analysis
    """
    with ExcelWriter(save_path) as writer:
        for iou_score_threshold_test, full_dict_dfs in full_dict_dfs_per_IoUscorethreshold.items():
            num_done = 0
            print('iou score thresh', iou_score_threshold_test)
            for model_checkpoint, dict_dfs in full_dict_dfs.items():
                for true_and_pred_matching_threshold, list_dfs in dict_dfs.items():
                    print('true/pred match tresh', true_and_pred_matching_threshold)
                    for sheet_name, df in zip(sheet_names, list_dfs):
                        print('on sheet name', sheet_name)
                        startrow_multiplier = df.shape[0]
                        if num_done == 0:
                            print(num_done, 0)
                            df.to_excel(writer, sheet_name+'_'+str(iou_score_threshold_test)+'_PredThresh', startrow=0)
                        else:
                            print(num_done, num_done*startrow_multiplier+2)
                            df.to_excel(writer, sheet_name+'_'+str(iou_score_threshold_test)+'_PredThresh', startrow=num_done*(startrow_multiplier+2))
                    num_done += 1
        writer.save()
    return

def save_excel_together_singlereport(list_dfs, sheet_names, save_path):
    with ExcelWriter(save_path) as writer:
        for sheet_name, df in zip(sheet_names, list_dfs):
             df.to_excel(writer, sheet_name)
        writer.save()
    return

def analysis_setup(test_dataset_path, input_yaml):
    for _, __, files in os.walk(test_dataset_path):
        validation_image_filenames = sorted(files)

    for _, __, files in os.walk(input_yaml['output_dir']):
        model_checkpoints_all = [f for f in files if '.pth' in f]
        for model_checkpoint in model_checkpoints_all:
            if 'final' in model_checkpoint:
                model_checkpoints_all.remove(model_checkpoint)

    # Get the subset of model checkpoints to analyze
    checkpoint_numbers_all = [int(ck.split('.pth')[0].split('_')[1]) for ck in model_checkpoints_all]

    num_checkpoints = input_yaml['num_checkpoints']
    if input_yaml['num_checkpoints'] > len(model_checkpoints_all):
        print('Setting of num_checkpoints greater than number of model checkpoints, resetting to be all checkpoints in output path')
        num_checkpoints = len(model_checkpoints_all)

    if input_yaml['only_last_checkpoint'] == True:
        #checkpoint_numbers = [int(ck.split('.pth')[0].split('_')[1]) for ck in model_checkpoints]
        only_checkpoint_number = max(checkpoint_numbers_all)
        for model_checkpoint in model_checkpoints_all:
            if str(only_checkpoint_number) in model_checkpoint:
                only_checkpoint_name = model_checkpoint
        model_checkpoints = list()
        model_checkpoints.append(only_checkpoint_name)
        num_checkpoints = 1
        print('Only analyzing last checkpoint', only_checkpoint_name)
    else:
        model_checkpoints = list()
        for checkpoint_number in input_yaml['checkpoint_numbers']:
            for model_checkpoint in model_checkpoints_all:
                # TODO: have better logic here. Sometimes string matching will pick wrong checkpoint
                if str(checkpoint_number) in model_checkpoint:
                    model_checkpoints.append(model_checkpoint)

    num_images = input_yaml['num_images']
    if input_yaml['num_images'] > len(validation_image_filenames):
        print('Setting of num_images greater than number of validation images, resetting to be all validation images')
        num_images = len(validation_image_filenames)

    return validation_image_filenames, model_checkpoints, num_checkpoints, num_images

def plot_overall_stats_vs_iou_threshold(save_path, full_dict_dfs_per_IoUscorethreshold):
    # Need to parse the full_dict_dfs dictionary and get overall P, R, F1 as function of IoU pred threshold and true vs pred defect scoring
    IoUscorethresholds = list()
    TruevsPredthresholds = list()
    precisions_iouthresh = dict()
    recalls_iouthresh = dict()
    f1s_iouthresh = dict()
    precisions_truepredthresh = dict()
    recalls_truepredthresh = dict()
    f1s_truepredthresh = dict()
    for iou_score_threshold_test, full_dict_dfs in full_dict_dfs_per_IoUscorethreshold.items():
        if iou_score_threshold_test not in IoUscorethresholds:
            IoUscorethresholds.append(iou_score_threshold_test)
        for model_checkpoint, dict_dfs in full_dict_dfs.items():
            for true_and_pred_matching_threshold, list_dfs in dict_dfs.items():
                overall_stats = list_dfs[0]
                if iou_score_threshold_test not in precisions_iouthresh.keys():
                    precisions_iouthresh[iou_score_threshold_test]={true_and_pred_matching_threshold : float(np.array(overall_stats['overall precision'])[0])}
                    recalls_iouthresh[iou_score_threshold_test]={true_and_pred_matching_threshold: float(np.array(overall_stats['overall recall'])[0])}
                    f1s_iouthresh[iou_score_threshold_test]={true_and_pred_matching_threshold: float(np.array(overall_stats['overall F1'])[0])}
                else:
                    precisions_iouthresh[iou_score_threshold_test].update({true_and_pred_matching_threshold : float(np.array(overall_stats['overall precision'])[0])})
                    recalls_iouthresh[iou_score_threshold_test].update({true_and_pred_matching_threshold: float(np.array(overall_stats['overall recall'])[0])})
                    f1s_iouthresh[iou_score_threshold_test].update({true_and_pred_matching_threshold: float(np.array(overall_stats['overall F1'])[0])})
                if true_and_pred_matching_threshold not in precisions_truepredthresh.keys():
                    precisions_truepredthresh[true_and_pred_matching_threshold] = {iou_score_threshold_test: float(np.array(overall_stats['overall precision'])[0])}
                    recalls_truepredthresh[true_and_pred_matching_threshold] = {iou_score_threshold_test: float(np.array(overall_stats['overall recall'])[0])}
                    f1s_truepredthresh[true_and_pred_matching_threshold] = {iou_score_threshold_test: float(np.array(overall_stats['overall F1'])[0])}
                else:
                    precisions_truepredthresh[true_and_pred_matching_threshold].update({iou_score_threshold_test: float(np.array(overall_stats['overall precision'])[0])})
                    recalls_truepredthresh[true_and_pred_matching_threshold].update({iou_score_threshold_test: float(np.array(overall_stats['overall recall'])[0])})
                    f1s_truepredthresh[true_and_pred_matching_threshold].update({iou_score_threshold_test: float(np.array(overall_stats['overall F1'])[0])})
                if true_and_pred_matching_threshold not in TruevsPredthresholds:
                    TruevsPredthresholds.append(true_and_pred_matching_threshold)

    for iou_score_threshold_test in IoUscorethresholds:
        precisions_list = [v for v in precisions_iouthresh[iou_score_threshold_test].values()]
        recalls_list = [v for v in recalls_iouthresh[iou_score_threshold_test].values()]
        f1s_list = [v for v in f1s_iouthresh[iou_score_threshold_test].values()]
        fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)

        ax.plot(TruevsPredthresholds, precisions_list, color='red', marker='o', markersize= 12, linestyle='solid', linewidth=1.5, label='Precision')
        ax.plot(TruevsPredthresholds, recalls_list, color='green', marker='^', markersize= 12, linestyle='solid', linewidth=1.5, label='Recall')
        ax.plot(TruevsPredthresholds, f1s_list, color='blue', marker='H', markersize= 12, linestyle='solid', linewidth=1.5, label='F1')

        ax.set_xlabel('IoU Test vs. Predicted box threshold', fontsize=16)
        ax.set_ylabel('Performance metric', fontsize=16)
        ax.set_ylim(bottom=0.0, top=1.0)
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_fontsize(14)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_fontsize(14)
        ax.legend(loc='best', fontsize=14, frameon=True)
        fig.savefig(os.path.join(save_path, 'Overall_Stats_vs_IoU_TestvsPredBox_threshold_'+str(iou_score_threshold_test)+'_IoUPredThreshold'+'.png'), dpi=200, bbox_inches='tight')

    for true_and_pred_matching_threshold in TruevsPredthresholds:
        precisions_list = [v for v in precisions_truepredthresh[true_and_pred_matching_threshold].values()]
        recalls_list = [v for v in recalls_truepredthresh[true_and_pred_matching_threshold].values()]
        f1s_list = [v for v in f1s_truepredthresh[true_and_pred_matching_threshold].values()]
        fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)

        ax.plot(IoUscorethresholds, precisions_list, color='red', marker='o', markersize= 12, linestyle='solid', linewidth=1.5, label='Precision')
        ax.plot(IoUscorethresholds, recalls_list, color='green', marker='^', markersize= 12, linestyle='solid', linewidth=1.5, label='Recall')
        ax.plot(IoUscorethresholds, f1s_list, color='blue', marker='H', markersize= 12, linestyle='solid', linewidth=1.5, label='F1')

        ax.set_xlabel('IoU predictor test threshold', fontsize=16)
        ax.set_ylabel('Performance metric', fontsize=16)
        ax.set_ylim(bottom=0.0, top=1.0)
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_fontsize(14)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_fontsize(14)
        ax.legend(loc='best', fontsize=14, frameon=True)
        fig.savefig(os.path.join(save_path, 'Overall_Stats_vs_IoU_predictor_threshold_'+str(true_and_pred_matching_threshold)+'_TruevsPredBoxThreshold'+'.png'), dpi=200, bbox_inches='tight')

    return

def plot_learning_curve(cfg, classification_reports_all_checkpoints_train, classification_reports_all_checkpoints_test):

    P_100_train = list()
    P_111_train = list()
    P_bdot_train = list()
    R_100_train = list()
    R_111_train = list()
    R_bdot_train = list()
    F1_100_train = list()
    F1_111_train = list()
    F1_bdot_train = list()
    P_100_test = list()
    P_111_test = list()
    P_bdot_test = list()
    R_100_test = list()
    R_111_test = list()
    R_bdot_test = list()
    F1_100_test = list()
    F1_111_test = list()
    F1_bdot_test = list()
    checkpoints = list()

    # Collect train stats
    for checkpoint, stats in classification_reports_all_checkpoints_train.items():
        P_100_train.append(stats['100']['precision'])
        P_111_train.append(stats['111']['precision'])
        P_bdot_train.append(stats['bdot']['precision'])
        R_100_train.append(stats['100']['recall'])
        R_111_train.append(stats['111']['recall'])
        R_bdot_train.append(stats['bdot']['recall'])
        F1_100_train.append(stats['100']['f1-score'])
        F1_111_train.append(stats['111']['f1-score'])
        F1_bdot_train.append(stats['bdot']['f1-score'])
        checkpoints.append(int(checkpoint.split('_')[1]))

    # Collect test stats
    for checkpoint, stats in classification_reports_all_checkpoints_test.items():
        P_100_test.append(stats['100']['precision'])
        P_111_test.append(stats['111']['precision'])
        P_bdot_test.append(stats['bdot']['precision'])
        R_100_test.append(stats['100']['recall'])
        R_111_test.append(stats['111']['recall'])
        R_bdot_test.append(stats['bdot']['recall'])
        F1_100_test.append(stats['100']['f1-score'])
        F1_111_test.append(stats['111']['f1-score'])
        F1_bdot_test.append(stats['bdot']['f1-score'])

    P_all_train = list()
    R_all_train = list()
    F1_all_train = list()
    P_all_test = list()
    R_all_test = list()
    F1_all_test = list()
    for P1, P2, P3 in zip(P_100_train, P_111_train, P_bdot_train):
        P_all_train.append((P1 + P2 + P3) / 3)
    for R1, R2, R3 in zip(R_100_train, R_111_train, R_bdot_train):
        R_all_train.append((R1 + R2 + R3) / 3)
    for F1, F2, F3 in zip(F1_100_train, F1_111_train, F1_bdot_train):
        F1_all_train.append((F1 + F2 + F3) / 3)
    for P1, P2, P3 in zip(P_100_test, P_111_test, P_bdot_test):
        P_all_test.append((P1 + P2 + P3) / 3)
    for R1, R2, R3 in zip(R_100_test, R_111_test, R_bdot_test):
        R_all_test.append((R1 + R2 + R3) / 3)
    for F1, F2, F3 in zip(F1_100_test, F1_111_test, F1_bdot_test):
        F1_all_test.append((F1 + F2 + F3) / 3)

    # Plot the data and save plot
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.plot(checkpoints, P_all_train, color='red', marker='o', linestyle='dashed', linewidth=1, alpha=0.6, label='Train')
    ax.plot(checkpoints, P_all_test, color='red', marker='o', linewidth=1, label='Test')
    ax.set_xlabel('Model iterations', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.legend(loc='upper right', fontsize=14, frameon=True)
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'LearningCurve_Precision.png'), dpi=200, bbox_inches='tight')

    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.plot(checkpoints, R_all_train, color='green', marker='^', linestyle='dashed', linewidth=1, alpha=0.6, label='Train')
    ax.plot(checkpoints, R_all_test, color='green', marker='^', linewidth=1, label='Test')
    ax.set_xlabel('Model iterations', fontsize=18)
    ax.set_ylabel('Recall', fontsize=18)
    ax.legend(loc='upper right', fontsize=14, frameon=True)
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'LearningCurve_Recall.png'), dpi=200, bbox_inches='tight')

    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.plot(checkpoints, F1_all_train, color='blue', marker='H', linestyle='dashed', linewidth=1, alpha=0.6, label='Train')
    ax.plot(checkpoints, F1_all_test, color='blue', marker='H', linewidth=1, label='Test')
    ax.set_xlabel('Model iterations', fontsize=18)
    ax.set_ylabel('F1 score', fontsize=18)
    ax.legend(loc='upper right', fontsize=14, frameon=True)
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'LearningCurve_F1score.png'), dpi=200, bbox_inches='tight')

    return

def save_objecttotals_per_image(save_path, image_name, true_classes, pred_classes,
                                iou_score_threshold_test, true_and_pred_matching_threshold):
    # Make dict and save file of true and pred object totals
    true_count_111 = 0
    true_count_100 = 0
    true_count_bd = 0
    pred_count_111 = 0
    pred_count_100 = 0
    pred_count_bd = 0
    for true_class in true_classes:
        if true_class == 0: #bd
            true_count_bd += 1
        if true_class == 1: #111
            true_count_111 += 1
        if true_class == 2: #100
            true_count_100 += 1
    for pred_class in pred_classes:
        if pred_class == 0: #bd
            pred_count_bd += 1
        if pred_class == 1: #111
            pred_count_111 += 1
        if pred_class == 2: #100
            pred_count_100 += 1
    total_true_objects = true_count_100+true_count_111+true_count_bd
    total_pred_objects = pred_count_100+pred_count_111+pred_count_bd
    data_dict = {'true_count_111': true_count_111,
                 'true_count_bd': true_count_bd,
                 'true_count_100': true_count_100,
                 'total_true_objects': total_true_objects,
                 'pred_count_111': pred_count_111,
                 'pred_count_bd': pred_count_bd,
                 'pred_count_100': pred_count_100,
                 'total_pred_objects': total_pred_objects}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index')
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectTotals.xlsx'))
    return total_true_objects, total_pred_objects

def save_foundobjecttotals_per_image(save_path, image_name, num_found, total_true_objects, total_pred_objects,
                                     iou_score_threshold_test, true_and_pred_matching_threshold):
    # Make dict and save file of total found objects in image, regardless if defect type is correct
    fp = total_pred_objects - num_found
    fn = total_true_objects - num_found
    prec = num_found / (num_found + fp)
    recall = num_found / (num_found + fn)
    try:
        f1 = (2 * prec * recall) / (prec + recall)
    except:
        f1 = np.nan
    data_dict = {'total_true_objects': total_true_objects,
                 'total_pred_objects': total_pred_objects,
                 'total_found_objects': num_found,
                 'precision': prec,
                 'recall': recall,
                 'f1': f1}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index')
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_FoundObjectTotals.xlsx'))
    return f1

def save_foundcorrectobjects_per_image(save_path, image_name, t_bdot_p_bdot, t_111_p_bdot, t_100_p_bdot,
                                t_bdot_p_111, t_111_p_111, t_100_p_111, t_bdot_p_100, t_111_p_100, t_100_p_100,
                                       iou_score_threshold_test, true_and_pred_matching_threshold):
    cm_arr = np.array([[t_111_p_111, t_bdot_p_111, t_100_p_111],
                       [t_111_p_bdot, t_bdot_p_bdot, t_100_p_bdot],
                       [t_111_p_100, t_bdot_p_100, t_100_p_100]])
    df = pd.DataFrame(data=cm_arr, columns=['True 111', 'True bdot', 'True 100'],
                      index=['Pred 111', 'Pred bdot', 'Pred 100'])
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_FoundCorrectObjects.xlsx'))

    try:
        prec_bd = t_bdot_p_bdot/(t_bdot_p_bdot+t_111_p_bdot+t_100_p_bdot)
    except:
        prec_bd = np.nan
    try:
        prec_111 = t_111_p_111/(t_111_p_111+t_bdot_p_111+t_100_p_111)
    except:
        prec_111 = np.nan
    try:
        prec_100 = t_100_p_100 / (t_100_p_100 + t_bdot_p_100 + t_111_p_100)
    except:
        prec_100 = np.nan
    try:
        recall_bd = t_bdot_p_bdot/(t_bdot_p_bdot+t_bdot_p_111+t_bdot_p_100)
    except:
        recall_bd = np.nan
    try:
        recall_111 = t_111_p_111/(t_111_p_111+t_111_p_bdot+t_111_p_100)
    except:
        recall_111 = np.nan
    try:
        recall_100 = t_100_p_100 / (t_100_p_100 + t_100_p_bdot + t_100_p_111)
    except:
        recall_100 = np.nan
    try:
        f1_bd = (2*prec_bd*recall_bd)/(prec_bd+recall_bd)
    except:
        f1_bd = np.nan
    try:
        f1_111 = (2 * prec_111 * recall_111) / (prec_111 + recall_111)
    except:
        f1_111 = np.nan
    try:
        f1_100 = (2 * prec_100 * recall_100) / (prec_100 + recall_100)
    except:
        f1_100 = np.nan
    f1_avg = np.mean([f1_bd, f1_111, f1_100])
    return f1_avg

def save_objectdensities_per_image(save_path, image_name, true_classes, pred_classes, iou_score_threshold_test,
                                   true_and_pred_matching_threshold):
    # Make dict and save file of true and pred object totals
    true_count_111 = 0
    true_count_100 = 0
    true_count_bd = 0
    pred_count_111 = 0
    pred_count_100 = 0
    pred_count_bd = 0
    for true_class in true_classes:
        if true_class == 0: #bd
            true_count_bd += 1
        if true_class == 1: #111
            true_count_111 += 1
        if true_class == 2: #100
            true_count_100 += 1
    for pred_class in pred_classes:
        if pred_class == 0: #bd
            pred_count_bd += 1
        if pred_class == 1: #111
            pred_count_111 += 1
        if pred_class == 2: #100
            pred_count_100 += 1

    nm_per_pixel_70kx = 0.880523
    nm_per_pixel_100kx = 0.869141
    nm_per_pixel_300kx = 0.478516
    nm_per_pixel_500kx = 0.283203
    nm_per_pixel_onzone = 0.1416015625  # 2048x2048
    m_per_nm = 10**-9
    if "70" in image_name:
        real_area = (nm_per_pixel_70kx * 1024) ** 2
    elif "100" in image_name:
        real_area = (nm_per_pixel_100kx * 1024) ** 2
    elif "300" in image_name:
        real_area = (nm_per_pixel_300kx * 1024) ** 2
    elif "500" in image_name:
        real_area = (nm_per_pixel_500kx * 1024) ** 2
    elif "dalong" in image_name:
        # dalong is same size as 500kx
        real_area = (nm_per_pixel_500kx * 1024) ** 2
    elif "onzone" in image_name:
        real_area = (nm_per_pixel_onzone * 2048) ** 2
    else:
        print('Could not process image file name for density calculation')
        exit()
    true_density_111 = true_count_111 / real_area
    true_density_100 = true_count_100 / real_area
    true_density_bd = true_count_bd / real_area
    pred_density_111 = pred_count_111 / real_area
    pred_density_100 = pred_count_100 / real_area
    pred_density_bd = pred_count_bd / real_area
    percent_density_error_bd = 100*(abs(np.mean(true_density_bd)-np.mean(pred_density_bd))/np.mean(true_density_bd))
    percent_density_error_111 = 100*(abs(np.mean(true_density_111)-np.mean(pred_density_111))/np.mean(true_density_111))
    percent_density_error_100 = 100*(abs(np.mean(true_density_100)-np.mean(pred_density_100))/np.mean(true_density_100))
    data_dict = {'image area (nm^2)': real_area, 'true_count_bd': true_count_bd, 'true_density_bd': true_density_bd,
                 'true_count_111': true_count_111, 'true_density_111': true_density_111, 'true_count_100': true_count_100,
                 'true_density_100': true_density_100, 'pred_count_bd': pred_count_bd, 'pred_density_bd': pred_density_bd,
                 'pred_count_111': pred_count_111, 'pred_density_111': pred_density_111, 'pred_count_100': pred_count_100,
                 'pred_density_100': pred_density_100,
                 'percent_density_error_bd': percent_density_error_bd,
                 'percent_density_error_111': percent_density_error_111,
                 'percent_density_error_100': percent_density_error_100}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index').T
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectDensities.xlsx'))
    return percent_density_error_bd, percent_density_error_111, percent_density_error_100,\
           true_density_bd, true_density_111, true_density_100,\
           pred_density_bd, pred_density_111, pred_density_100

def save_objectsizes_per_image(save_path, image_name, true_classes, pred_classes, true_segs, pred_segs,
                               iou_score_threshold_test, true_and_pred_matching_threshold, mask_on):
    true_bd_sizes = list()
    true_111_sizes = list()
    true_100_sizes = list()
    pred_bd_sizes = list()
    pred_111_sizes = list()
    pred_100_sizes = list()
    true_bd_shapes = list()
    true_111_shapes = list()
    true_100_shapes = list()
    pred_bd_shapes = list()
    pred_111_shapes = list()
    pred_100_shapes = list()
    for seg, defect in zip(true_segs, true_classes):
        if mask_on == True:
            defect_size, defect_shape_factor = get_defect_size(segmentation=seg,
                                      image_name=image_name,
                                      defect_type=defect)
        else:
            defect_size = 0
            defect_shape_factor = 0

        if defect == 0: #bd
            true_bd_sizes.append(defect_size)
            true_bd_shapes.append(defect_shape_factor)
        elif defect == 1: #111
            true_111_sizes.append(defect_size)
            true_111_shapes.append(defect_shape_factor)
        elif defect == 2: #100
            true_100_sizes.append(defect_size)
            true_100_shapes.append(defect_shape_factor)
    for seg, defect in zip(pred_segs, pred_classes):
        if mask_on == True:
            defect_size, defect_shape_factor = get_defect_size(segmentation=seg,
                                      image_name=image_name,
                                      defect_type=defect)
        else:
            defect_size = 0
            defect_shape_factor = 0

        if defect == 0: #bd
            pred_bd_sizes.append(defect_size)
            pred_bd_shapes.append(defect_shape_factor)
        elif defect == 1: #111
            pred_111_sizes.append(defect_size)
            pred_111_shapes.append(defect_shape_factor)
        elif defect == 2: #100
            pred_100_sizes.append(defect_size)
            pred_100_shapes.append(defect_shape_factor)

    percent_size_error_bd = 100*(abs(np.mean(true_bd_sizes)-np.mean(pred_bd_sizes))/np.mean(true_bd_sizes))
    percent_size_error_111 = 100 * (abs(np.mean(true_111_sizes) - np.mean(pred_111_sizes)) / np.mean(true_111_sizes))
    percent_size_error_100 = 100 * (abs(np.mean(true_100_sizes) - np.mean(pred_100_sizes)) / np.mean(true_100_sizes))
    data_dict = {"true_sizes_bd": true_bd_sizes, "true_sizes_111": true_111_sizes, "true_sizes_100": true_100_sizes,
                 "pred_sizes_bd": pred_bd_sizes, "pred_sizes_111": pred_111_sizes, "pred_sizes_100": pred_100_sizes,
                 "true_avg_size_bd": [np.mean(true_bd_sizes)], "true_avg_size_111": [np.mean(true_111_sizes)], "true_avg_size_100": [np.mean(true_100_sizes)],
                 "true_stdev_size_bd": [np.std(true_bd_sizes)], "true_stdev_size_111": [np.std(true_111_sizes)], "true_stdev_size_100": [np.std(true_100_sizes)],
                 "pred_avg_size_bd": [np.mean(pred_bd_sizes)], "pred_avg_size_111": [np.mean(pred_111_sizes)], "pred_avg_size_100": [np.mean(pred_100_sizes)],
                 "pred_stdev_size_bd": [np.std(pred_bd_sizes)], "pred_stdev_size_111": [np.std(pred_111_sizes)], "pred_stdev_size_100": [np.std(pred_100_sizes)],
                 "true_shapes_bd": true_bd_shapes, "true_shapes_111": true_111_shapes, "true_shapes_100": true_100_shapes,
                 "pred_shapes_bd": pred_bd_shapes, "pred_shapes_111": pred_111_shapes, "pred_shapes_100": pred_100_shapes,
                 "true_avg_shape_bd": [np.mean(true_bd_shapes)], "true_avg_shape_111": [np.mean(true_111_shapes)], "true_avg_shape_100": [np.mean(true_100_shapes)],
                 "true_stdev_shape_bd": [np.std(true_bd_shapes)], "true_stdev_shape_111": [np.std(true_111_shapes)], "true_stdev_shape_100": [np.std(true_100_shapes)],
                 "pred_avg_shape_bd": [np.mean(pred_bd_shapes)], "pred_avg_shape_111": [np.mean(pred_111_shapes)], "pred_avg_shape_100": [np.mean(pred_100_shapes)],
                 "pred_stdev_shape_bd": [np.std(pred_bd_shapes)], "pred_stdev_shape_111": [np.std(pred_111_shapes)], "pred_stdev_shape_100": [np.std(pred_100_shapes)],
                 "percent_size_error_bd": [percent_size_error_bd],
                 "percent_size_error_111": [percent_size_error_111],
                 "percent_size_error_100": [percent_size_error_100]}
    df = pd.DataFrame.from_dict(data=data_dict, orient='index')
    #df = pd.DataFrame.from_dict(data=data_dict, orient='columns').T
    df.to_excel(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes.xlsx'))
    #####
    #
    # Make histograms of distribution of true and pred sizes for each defect type
    #
    #####
    # Histogram of bdot sizes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_bd_sizes, bins=np.arange(0, 20, 2), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_bd_sizes, bins=np.arange(0, 20, 2), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Black dot sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_bd_sizes), range(len(true_bd_sizes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true black dots', fontsize=12)
    ax2.plot(sorted(pred_bd_sizes), range(len(pred_bd_sizes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted black dots', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_bd_sizes)
        pred_skew = skew(pred_bd_sizes)
        true_stats = pd.DataFrame(true_bd_sizes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_bd_sizes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_Blackdot'+'.png'),
                dpi=250, bbox_inches='tight')

    # Histogram of 111 sizes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_111_sizes, bins=np.arange(0, 100, 10), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_111_sizes, bins=np.arange(0, 100, 10), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<111> loop sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_111_sizes), range(len(true_111_sizes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true <111> loops', fontsize=12)
    ax2.plot(sorted(pred_111_sizes), range(len(pred_111_sizes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted <111> loops', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_111_sizes)
        pred_skew = skew(pred_111_sizes)
        true_stats = pd.DataFrame(true_111_sizes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_111_sizes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_111loops'+'.png'),
                dpi=250, bbox_inches='tight')

    # Histogram of 100 sizes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_100_sizes, bins=np.arange(0, 100, 10), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_100_sizes, bins=np.arange(0, 100, 10), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<100> loop sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_100_sizes), range(len(true_100_sizes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true <100> loops', fontsize=12)
    ax2.plot(sorted(pred_100_sizes), range(len(pred_100_sizes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted <100> loops', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_100_sizes)
        pred_skew = skew(pred_100_sizes)
        true_stats = pd.DataFrame(true_100_sizes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_100_sizes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_100loops'+'.png'),
                dpi=250, bbox_inches='tight')

    # Histogram of all defect sizes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    all_true = true_111_sizes+true_bd_sizes+true_100_sizes
    all_pred = pred_111_sizes+pred_bd_sizes+pred_100_sizes
    ax.hist(all_true, bins=np.arange(0, 100, 10), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(all_pred, bins=np.arange(0, 100, 10), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Defect sizes (nm)', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(all_true), range(len(all_true)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true defects', fontsize=12)
    ax2.plot(sorted(all_pred), range(len(all_pred)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted defects', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(all_true)
        pred_skew = skew(all_pred)
        true_stats = pd.DataFrame(all_true).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(all_pred).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectSizes_AllDefects'+'.png'),
                dpi=250, bbox_inches='tight')

    #####
    #
    # Make histograms of distribution of true and pred shapes for each defect type
    #
    #####
    # Histogram of bdot shapes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_bd_shapes, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_bd_shapes, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Black dot Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_bd_shapes), range(len(true_bd_shapes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true black dots', fontsize=12)
    ax2.plot(sorted(pred_bd_shapes), range(len(pred_bd_shapes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted black dots', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_bd_shapes)
        pred_skew = skew(pred_bd_shapes)
        true_stats = pd.DataFrame(true_bd_shapes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_bd_shapes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_Blackdot'+'.png'),
                dpi=250, bbox_inches='tight')

    # Histogram of 111 shapes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_111_shapes, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_111_shapes, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<111> loop Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_111_shapes), range(len(true_111_shapes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true <111> loops', fontsize=12)
    ax2.plot(sorted(pred_111_shapes), range(len(pred_111_shapes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted <111> loops', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_111_shapes)
        pred_skew = skew(pred_111_shapes)
        true_stats = pd.DataFrame(true_111_shapes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_111_shapes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_111loops'+'.png'),
                dpi=250, bbox_inches='tight')

    # Histogram of 100 shapes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(true_100_shapes, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(pred_100_shapes, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('<100> loop Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(true_100_shapes), range(len(true_100_shapes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true <100> loops', fontsize=12)
    ax2.plot(sorted(pred_100_shapes), range(len(pred_100_shapes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted <100> loops', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(true_100_shapes)
        pred_skew = skew(pred_100_shapes)
        true_stats = pd.DataFrame(true_100_shapes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(pred_100_shapes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_100loops'+'.png'),
                dpi=250, bbox_inches='tight')

    # Histogram of all defect shapes
    all_true_shapes = true_100_shapes+true_111_shapes+true_bd_shapes
    all_pred_shapes = pred_100_shapes+pred_111_shapes+pred_bd_shapes
    fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
    ax.hist(all_true_shapes, bins=np.arange(1, 2, 0.1), color='b', edgecolor='k', alpha=0.50, label='True')
    ax.hist(all_pred_shapes, bins=np.arange(1, 2, 0.1), color='g', edgecolor='k', alpha=0.50, label='Predicted')
    ax.set_xlabel('Defect Heywood circularity', fontsize=12)
    ax.set_ylabel('Number of instances', fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(sorted(all_true_shapes), range(len(all_true_shapes)), 'b--', linewidth=1)
    ax2.set_ylabel('Total number of true defects', fontsize=12)
    ax2.plot(sorted(all_pred_shapes), range(len(all_pred_shapes)), 'g--', linewidth=1)
    ax2.set_ylabel('Total number of predicted defects', fontsize=12)
    ax.legend(loc='lower right')
    try:
        true_skew = skew(all_true_shapes)
        pred_skew = skew(all_pred_shapes)
        true_stats = pd.DataFrame(all_true_shapes).describe().to_dict()[0]
        true_stats['skew'] = true_skew
        pred_stats = pd.DataFrame(all_pred_shapes).describe().to_dict()[0]
        pred_stats['skew'] = pred_skew
        plot_stats(fig, true_stats, x_align=0.725, y_align=0.90, type='float', font_dict={'color': 'b'})
        plot_stats(fig, pred_stats, x_align=0.725, y_align=0.50, type='float', font_dict={'color': 'g'})
    except:
        pass
    fig.savefig(os.path.join(save_path, image_name+'_TruePredMatch_'+str(true_and_pred_matching_threshold)+
                             '_IoUScoreThresh_'+str(iou_score_threshold_test)+'_TruePred_ObjectShapes_AllDefects'+'.png'),
                dpi=250, bbox_inches='tight')

    return percent_size_error_bd, percent_size_error_111, percent_size_error_100, \
           np.mean(true_bd_sizes), np.mean(true_111_sizes), np.mean(true_100_sizes),\
           np.mean(pred_bd_sizes), np.mean(pred_111_sizes), np.mean(pred_100_sizes),\
           np.mean(true_bd_shapes), np.mean(true_111_shapes), np.mean(true_100_shapes),\
           np.mean(pred_bd_shapes), np.mean(pred_111_shapes), np.mean(pred_100_shapes), \
           true_bd_sizes, true_111_sizes, true_100_sizes,\
           pred_bd_sizes, pred_111_sizes, pred_100_sizes,\
           true_bd_shapes, true_111_shapes, true_100_shapes,\
           pred_bd_shapes, pred_111_shapes, pred_100_shapes

def analyze_checkpoints(cfg, defect_metadata, input_yaml, iou_score_threshold_test, test_dataset_path,
                        anno_dict_list_val, file_note, only_last_checkpoint=True, true_and_pred_matching_method='iou_bbox',
                        true_and_pred_matching_thresholds=[0.7]):

    '''
    true_and_pred_matching_method (str): must be one of "iou_bbox" and "pixelnorm_mask"
    true_and_pred_matching_thresholds (list): values of thresholds used to determine if true and predicted masks/bboxes
        correspond to the same defect. Values represent bbox IoUs and must be <=1 and >=0 if true_and_pred_matching_method=='iou_bbox',
        or values represent the norm of a pixel distance (typically values may be from 30 to 80) if true_and_pred_matching_method=='pixelnorm_mask'
    '''

    # Whether to save all pixel-wise data to excel sheets for each image
    save_all_data = False
    save_images = True
    save_image_classification_reports = False

    validation_image_filenames, model_checkpoints, num_checkpoints, num_images = analysis_setup(test_dataset_path=test_dataset_path,
                                                                                                input_yaml=input_yaml)

    classification_reports_all_checkpoints_pixels = dict()
    dict_dfs = dict()
    full_dict_dfs = dict()
    checkpoints_done = 0
    for model_checkpoint in model_checkpoints:
        print('ANALYZING MODEL CHECKPOINT ', model_checkpoint)

        if checkpoints_done < num_checkpoints:
            # Here- loop over values of threshold to discern true and predicted masks/bboxes
            for true_and_pred_matching_threshold in true_and_pred_matching_thresholds:

                images_done = 0
                true_pixels_all = list()
                pred_pixels_all = list()
                true_classes_all = list()
                pred_classes_all = list()
                t_bdot_p_bdot = 0
                t_111_p_bdot = 0
                t_100_p_bdot = 0
                t_bdot_p_111 = 0
                t_111_p_111 = 0
                t_100_p_111 = 0
                t_bdot_p_100 = 0
                t_111_p_100 = 0
                t_100_p_100 = 0

                # Values needed to get overall TP, FP, FN values
                num_true_perimage = list()
                num_pred_perimage = list()
                num_found_perimage = list()

                true_defectsizes_bdot_nm_foundonly = list()
                true_defectsizes_111_nm_foundonly = list()
                true_defectsizes_100_nm_foundonly = list()
                pred_defectsizes_bdot_nm_foundonly = list()
                pred_defectsizes_111_nm_foundonly = list()
                pred_defectsizes_100_nm_foundonly = list()
                true_defectshapes_bdot_foundonly = list()
                true_defectshapes_111_foundonly = list()
                true_defectshapes_100_foundonly = list()
                pred_defectshapes_bdot_foundonly = list()
                pred_defectshapes_111_foundonly = list()
                pred_defectshapes_100_foundonly = list()

                true_defectsizes_bdot_nm_all = list()
                true_defectsizes_111_nm_all = list()
                true_defectsizes_100_nm_all = list()
                pred_defectsizes_bdot_nm_all = list()
                pred_defectsizes_111_nm_all = list()
                pred_defectsizes_100_nm_all = list()
                true_defectshapes_bdot_all = list()
                true_defectshapes_111_all = list()
                true_defectshapes_100_all = list()
                pred_defectshapes_bdot_all = list()
                pred_defectshapes_111_all = list()
                pred_defectshapes_100_all = list()

                percent_density_error_bd_perimage_list = list()
                percent_density_error_111_perimage_list = list()
                percent_density_error_100_perimage_list = list()
                percent_size_error_bd_perimage_list = list()
                percent_size_error_111_perimage_list = list()
                percent_size_error_100_perimage_list = list()
                true_density_bd_perimage_list = list()
                true_density_111_perimage_list = list()
                true_density_100_perimage_list = list()
                pred_density_bd_perimage_list = list()
                pred_density_111_perimage_list = list()
                pred_density_100_perimage_list = list()
                true_avg_size_bd_perimage_list = list()
                true_avg_size_111_perimage_list = list()
                true_avg_size_100_perimage_list = list()
                pred_avg_size_bd_perimage_list = list()
                pred_avg_size_111_perimage_list = list()
                pred_avg_size_100_perimage_list = list()
                true_avg_shape_bd_perimage_list = list()
                true_avg_shape_111_perimage_list = list()
                true_avg_shape_100_perimage_list = list()
                pred_avg_shape_bd_perimage_list = list()
                pred_avg_shape_111_perimage_list = list()
                pred_avg_shape_100_perimage_list = list()
                image_name_list = list()

                data_dict_per_image = dict()

                for filename in validation_image_filenames:
                    if images_done < num_images:
                    #if (images_done < num_images) and (filename not in ['K713_300kx_store4_grid1_0005.jpg','K713_300kx_store4_grid1_0011.jpg']): # This image caused trouble in predictions for some reason
                        if filename not in list(data_dict_per_image.keys()):
                            data_dict_per_image[filename] = dict()

                        num_found = 0

                        predictor = get_predictor(cfg=cfg, model_checkpoint=model_checkpoint,
                                                  iou_score_threshold_test=iou_score_threshold_test,
                                                  test_dataset_path=test_dataset_path)

                        true_pixels_all_oneimage, true_classes_all_oneimage, true_segmentations_oneimage, \
                        true_boxes_oneimage = get_true_data_stats(cfg=cfg, defect_metadata=defect_metadata,
                                                                anno_dict_list_val=anno_dict_list_val,
                                                                filename=filename, model_checkpoint=model_checkpoint,
                                                                true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                iou_score_threshold_test=iou_score_threshold_test, show_images=False,
                                                                save_images=save_images, save_all_data=save_all_data,
                                                                mask_on=input_yaml['mask_on'])

                        #print('TRUE BOXES, ONE IMAGE')
                        #print(true_boxes_oneimage)

                        pred_pixels_all_oneimage, pred_classes_all_oneimage, pred_segmentations_oneimage, \
                        pred_boxes_oneimage = get_pred_data_stats(cfg=cfg, defect_metadata=defect_metadata,
                                                                anno_dict_list_val=anno_dict_list_val,
                                                                filename=filename, predictor=predictor,
                                                                model_checkpoint=model_checkpoint,
                                                                true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                iou_score_threshold_test=iou_score_threshold_test,
                                                                show_images=False, save_images=save_images,
                                                                save_all_data=save_all_data,
                                                                mask_on=input_yaml['mask_on'])

                        #print('PRED BOXES, ONE IMAGE')
                        #print(pred_boxes_oneimage)

                        print('SHAPES OF PRED BOXES, CLASSES, SEGS')
                        print(len(pred_boxes_oneimage), len(pred_classes_all_oneimage), len(pred_segmentations_oneimage))

                        # TODO: Don't need to sort anymore as use IoU to find correct defect positions
                        #pred_boxes_oneimage_sorted = pred_boxes_oneimage
                        #pred_classes_all_oneimage_sorted = pred_classes_all_oneimage
                        #pred_segmentations_oneimage_sorted = pred_segmentations_oneimage
                        #true_boxes_oneimage_sorted = true_boxes_oneimage
                        #true_classes_all_oneimage_sorted = true_classes_all_oneimage
                        #true_segmentations_oneimage_sorted = true_segmentations_oneimage

                        #TODO: consider sorting on bounding box values so can work with both Faster and Mask R-CNN models
                        true_segmentations_oneimage_sorted, true_classes_all_oneimage_sorted, true_boxes_oneimage_sorted = \
                            (list(t) for t in zip(*sorted(zip(true_segmentations_oneimage, true_classes_all_oneimage, true_boxes_oneimage))))
                        true_segmentations_oneimage_abbrev = list()
                        for i, true_seg in enumerate(true_segmentations_oneimage_sorted):
                            true_segmentations_oneimage_abbrev.append([true_seg[0][0], true_seg[1][0]])

                        # TODO: consider sorting on bounding box values so can work with both Faster and Mask R-CNN models
                        pred_segmentations_oneimage_sorted, pred_classes_all_oneimage_sorted, pred_boxes_oneimage_sorted = \
                            (list(t) for t in zip(*sorted(zip(pred_segmentations_oneimage, pred_classes_all_oneimage, pred_boxes_oneimage))))
                        pred_segmentations_oneimage_abbrev = list()
                        for i, pred_seg in enumerate(pred_segmentations_oneimage_sorted):
                            pred_segmentations_oneimage_abbrev.append([pred_seg[0][0], pred_seg[1][0]])

                        # Here- using norm of pixel distance to match up true and predicted masks.
                        if true_and_pred_matching_method == 'pixelnorm_mask':
                            num_found, t_111_p_111, t_111_p_bdot, t_111_p_100, t_bdot_p_111, t_bdot_p_bdot, t_bdot_p_100, \
                            t_100_p_111, t_100_p_bdot, t_100_p_100, true_defectsizes_bdot_nm_foundonly, true_defectsizes_111_nm_foundonly, \
                            true_defectsizes_100_nm_foundonly, pred_defectsizes_bdot_nm_foundonly, pred_defectsizes_111_nm_foundonly, pred_defectsizes_100_nm_foundonly = \
                                match_true_and_predicted_defects_pixelnorm_mask(true_segmentations_oneimage_abbrev, pred_segmentations_oneimage_abbrev,
                                                    true_classes_all_oneimage_sorted, pred_classes_all_oneimage_sorted,
                                                    true_segmentations_oneimage_sorted, pred_segmentations_oneimage_sorted,
                                                    num_found, t_111_p_111, t_111_p_bdot, t_111_p_100,
                                                    t_bdot_p_111, t_bdot_p_bdot, t_bdot_p_100,
                                                    t_100_p_111, t_100_p_bdot, t_100_p_100,
                                                    true_defectsizes_bdot_nm_foundonly, true_defectsizes_111_nm_foundonly, true_defectsizes_100_nm_foundonly,
                                                    pred_defectsizes_bdot_nm_foundonly, pred_defectsizes_111_nm_foundonly, pred_defectsizes_100_nm_foundonly,
                                                    filename)
                        # Here- using IoU of true and predicted bounding boxes to match true and predicted masks
                        elif true_and_pred_matching_method == 'iou_bbox':
                            num_found, t_111_p_111_oneimage, t_111_p_bdot_oneimage, t_111_p_100_oneimage, \
                            t_bdot_p_111_oneimage, t_bdot_p_bdot_oneimage, t_bdot_p_100_oneimage, \
                            t_100_p_111_oneimage, t_100_p_bdot_oneimage, t_100_p_100_oneimage, \
                            true_defectsizes_bdot_nm_foundonly, true_defectsizes_111_nm_foundonly, \
                            true_defectsizes_100_nm_foundonly, pred_defectsizes_bdot_nm_foundonly, pred_defectsizes_111_nm_foundonly, pred_defectsizes_100_nm_foundonly, \
                            true_defectshapes_bdot_foundonly, true_defectshapes_111_foundonly, true_defectshapes_100_foundonly, \
                            pred_defectshapes_bdot_foundonly, pred_defectshapes_111_foundonly, pred_defectshapes_100_foundonly = \
                                match_true_and_predicted_defects_iou_bbox(true_classes_all_oneimage_sorted, pred_classes_all_oneimage_sorted,
                                    true_segmentations_oneimage_sorted, pred_segmentations_oneimage_sorted,
                                    true_boxes_oneimage_sorted, pred_boxes_oneimage_sorted,
                                    num_found,
                                    true_defectsizes_bdot_nm_foundonly, true_defectsizes_111_nm_foundonly, true_defectsizes_100_nm_foundonly,
                                    pred_defectsizes_bdot_nm_foundonly, pred_defectsizes_111_nm_foundonly, pred_defectsizes_100_nm_foundonly,
                                    true_defectshapes_bdot_foundonly, true_defectshapes_111_foundonly, true_defectshapes_100_foundonly,
                                    pred_defectshapes_bdot_foundonly, pred_defectshapes_111_foundonly, pred_defectshapes_100_foundonly,
                                    image_name=filename, mask_on=input_yaml['mask_on'], iou_threshold =true_and_pred_matching_threshold)
                            t_bdot_p_bdot += t_bdot_p_bdot_oneimage
                            t_111_p_bdot += t_111_p_bdot_oneimage
                            t_100_p_bdot += t_100_p_bdot_oneimage
                            t_bdot_p_111 += t_bdot_p_111_oneimage
                            t_111_p_111 += t_111_p_111_oneimage
                            t_100_p_111 += t_100_p_111_oneimage
                            t_bdot_p_100 += t_bdot_p_100_oneimage
                            t_111_p_100 += t_111_p_100_oneimage
                            t_100_p_100 += t_100_p_100_oneimage
                        else:
                            raise ValueError("true_and_pred_matching_method must be one of 'pixelnorm_mask' or 'iou_bbox'")

                        # Here, append number of defects found, true number of defects and predicted number of defects for this particular image
                        print('FOUND ', num_found, 'CORRECT INSTANCES FOUND')
                        num_found_perimage.append(num_found)
                        num_true_perimage.append(len(true_boxes_oneimage))
                        num_pred_perimage.append(len(pred_boxes_oneimage))

                        ########
                        #
                        # HERE save key per-image stats files here
                        #
                        ########
                        image_name = filename[:-4]
                        total_true_objects, total_pred_objects = save_objecttotals_per_image(save_path=cfg.OUTPUT_DIR,
                                                                                             image_name=image_name,
                                                                                             true_classes=true_classes_all_oneimage,
                                                                                             pred_classes=pred_classes_all_oneimage,
                                                                                             iou_score_threshold_test=iou_score_threshold_test,
                                                                                             true_and_pred_matching_threshold=true_and_pred_matching_threshold)

                        f1 = save_foundobjecttotals_per_image(save_path=cfg.OUTPUT_DIR,
                                                         image_name=image_name,
                                                         num_found=num_found,
                                                         total_true_objects=total_true_objects,
                                                         total_pred_objects=total_pred_objects,
                                                         iou_score_threshold_test=iou_score_threshold_test,
                                                         true_and_pred_matching_threshold=true_and_pred_matching_threshold
                                                         )

                        data_dict_per_image[filename]['overall F1'] = f1
                        data_dict_per_image[filename]['total true objects'] = total_true_objects
                        data_dict_per_image[filename]['total pred objects'] = total_pred_objects

                        f1_found = save_foundcorrectobjects_per_image(save_path=cfg.OUTPUT_DIR,
                                                           image_name=image_name,
                                                           t_bdot_p_bdot=t_bdot_p_bdot_oneimage,
                                                           t_111_p_bdot=t_111_p_bdot_oneimage,
                                                           t_100_p_bdot=t_100_p_bdot_oneimage,
                                                           t_bdot_p_111=t_bdot_p_111_oneimage,
                                                           t_111_p_111=t_111_p_111_oneimage,
                                                           t_100_p_111=t_100_p_111_oneimage,
                                                           t_bdot_p_100=t_bdot_p_100_oneimage,
                                                           t_111_p_100=t_111_p_100_oneimage,
                                                           t_100_p_100=t_100_p_100_oneimage,
                                                           iou_score_threshold_test=iou_score_threshold_test,
                                                           true_and_pred_matching_threshold=true_and_pred_matching_threshold
                                                           )

                        data_dict_per_image[filename]['defect find F1'] = f1_found

                        percent_density_error_bd, percent_density_error_111, percent_density_error_100,\
                        true_density_bd, true_density_111, true_density_100,\
                        pred_density_bd, pred_density_111, pred_density_100 = save_objectdensities_per_image(save_path=cfg.OUTPUT_DIR,
                                                       image_name=image_name,
                                                       true_classes=true_classes_all_oneimage,
                                                       pred_classes=pred_classes_all_oneimage,
                                                       iou_score_threshold_test=iou_score_threshold_test,
                                                       true_and_pred_matching_threshold=true_and_pred_matching_threshold
                                                       )
                        percent_density_error_bd_perimage_list.append(percent_density_error_bd)
                        percent_density_error_111_perimage_list.append(percent_density_error_111)
                        percent_density_error_100_perimage_list.append(percent_density_error_100)
                        true_density_bd_perimage_list.append(true_density_bd)
                        true_density_111_perimage_list.append(true_density_111)
                        true_density_100_perimage_list.append(true_density_100)
                        pred_density_bd_perimage_list.append(pred_density_bd)
                        pred_density_111_perimage_list.append(pred_density_111)
                        pred_density_100_perimage_list.append(pred_density_100)

                        data_dict_per_image[filename]['true bd density'] = true_density_bd
                        data_dict_per_image[filename]['pred bd density'] = pred_density_bd
                        data_dict_per_image[filename]['true 111 density'] = true_density_111
                        data_dict_per_image[filename]['pred 111 density'] = pred_density_111
                        data_dict_per_image[filename]['true 100 density'] = true_density_100
                        data_dict_per_image[filename]['pred 100 density'] = pred_density_100
                        data_dict_per_image[filename]['avg density error'] = np.mean([percent_density_error_bd,
                                                                                      percent_density_error_111,
                                                                                      percent_density_error_100])

                        percent_size_error_bd, percent_size_error_111, percent_size_error_100, \
                        true_avg_size_bd, true_avg_size_111, true_avg_size_100, \
                        pred_avg_size_bd, pred_avg_size_111, pred_avg_size_100,\
                        true_avg_shape_bd, true_avg_shape_111, true_avg_shape_100,\
                        pred_avg_shape_bd, pred_avg_shape_111, pred_avg_shape_100, \
                        true_bd_sizes, true_111_sizes, true_100_sizes, \
                        pred_bd_sizes, pred_111_sizes, pred_100_sizes, \
                        true_bd_shapes, true_111_shapes, true_100_shapes,\
                        pred_bd_shapes, pred_111_shapes, pred_100_shapes = save_objectsizes_per_image(save_path=cfg.OUTPUT_DIR,
                                                   image_name=image_name,
                                                   true_classes=true_classes_all_oneimage,
                                                   pred_classes=pred_classes_all_oneimage,
                                                   true_segs=true_segmentations_oneimage,
                                                   pred_segs=pred_segmentations_oneimage,
                                                   iou_score_threshold_test=iou_score_threshold_test,
                                                   true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                   mask_on=input_yaml['mask_on']
                                                   )
                        percent_size_error_bd_perimage_list.append(percent_size_error_bd)
                        percent_size_error_111_perimage_list.append(percent_size_error_111)
                        percent_size_error_100_perimage_list.append(percent_size_error_100)
                        true_avg_size_bd_perimage_list.append(true_avg_size_bd)
                        true_avg_size_111_perimage_list.append(true_avg_size_111)
                        true_avg_size_100_perimage_list.append(true_avg_size_100)
                        pred_avg_size_bd_perimage_list.append(pred_avg_size_bd)
                        pred_avg_size_111_perimage_list.append(pred_avg_size_111)
                        pred_avg_size_100_perimage_list.append(pred_avg_size_100)
                        true_avg_shape_bd_perimage_list.append(true_avg_shape_bd)
                        true_avg_shape_111_perimage_list.append(true_avg_shape_111)
                        true_avg_shape_100_perimage_list.append(true_avg_shape_100)
                        pred_avg_shape_bd_perimage_list.append(pred_avg_shape_bd)
                        pred_avg_shape_111_perimage_list.append(pred_avg_shape_111)
                        pred_avg_shape_100_perimage_list.append(pred_avg_shape_100)

                        data_dict_per_image[filename]['true bd avg size'] = true_avg_size_bd
                        data_dict_per_image[filename]['pred bd avg size'] = pred_avg_size_bd
                        data_dict_per_image[filename]['true 111 avg size'] = true_avg_size_111
                        data_dict_per_image[filename]['pred 111 avg size'] = pred_avg_size_111
                        data_dict_per_image[filename]['true 100 avg size'] = true_avg_size_100
                        data_dict_per_image[filename]['pred 100 avg size'] = pred_avg_size_100
                        data_dict_per_image[filename]['avg size error'] = np.mean([percent_size_error_bd,
                                                                                  percent_size_error_111,
                                                                                  percent_size_error_100])
                        data_dict_per_image[filename]['true bd avg shape'] = true_avg_shape_bd
                        data_dict_per_image[filename]['pred bd avg shape'] = pred_avg_shape_bd
                        data_dict_per_image[filename]['true 111 avg shape'] = true_avg_shape_111
                        data_dict_per_image[filename]['pred 111 avg shape'] = pred_avg_shape_111
                        data_dict_per_image[filename]['true 100 avg shape'] = true_avg_shape_100
                        data_dict_per_image[filename]['pred 100 avg shape'] = pred_avg_shape_100

                        true_defectsizes_bdot_nm_all += true_bd_sizes
                        true_defectsizes_111_nm_all += true_111_sizes
                        true_defectsizes_100_nm_all += true_100_sizes
                        true_defectshapes_bdot_all += true_bd_shapes
                        true_defectshapes_111_all += true_111_shapes
                        true_defectshapes_100_all += true_100_shapes
                        pred_defectsizes_bdot_nm_all += pred_bd_sizes
                        pred_defectsizes_111_nm_all += pred_111_sizes
                        pred_defectsizes_100_nm_all += pred_100_sizes
                        pred_defectshapes_bdot_all += pred_bd_shapes
                        pred_defectshapes_111_all += pred_111_shapes
                        pred_defectshapes_100_all += pred_100_shapes


                        #if save_image_classification_reports == True:
                        #    report_asdict = get_pixel_classification_report(cfg=cfg, filename=filename,
                        #                                              true_pixels_all=true_pixels_all_oneimage,
                        #                                              pred_pixels_all=pred_pixels_all_oneimage)
                        true_pixels_all.append(true_pixels_all_oneimage.flatten().tolist())
                        pred_pixels_all.append(pred_pixels_all_oneimage.flatten().tolist())
                        true_classes_all.append(true_classes_all_oneimage)
                        pred_classes_all.append(pred_classes_all_oneimage)

                        image_name_list.append(filename)
                        images_done += 1

                # Now get classification report for all analyzed images together
                true_pixels_all_flattened = np.array(true_pixels_all).flatten()
                pred_pixels_all_flattened = np.array(pred_pixels_all).flatten()
                # Make list of lists into one long list
                true_classes_all_flattened = list(itertools.chain(*true_classes_all))
                pred_classes_all_flattened = list(itertools.chain(*pred_classes_all))


                # Total up the number of instances that are true, predicted, and found correctly for overall P, R, F1 scores
                df_overallstats = get_overall_defect_stats(num_true_perimage=num_true_perimage,
                                                           num_pred_perimage=num_pred_perimage,
                                                           num_found_perimage=num_found_perimage,
                                                           model_checkpoint =model_checkpoint,
                                                           iou_score_threshold_test=iou_score_threshold_test,
                                                           true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                           save_path=os.path.join(cfg.OUTPUT_DIR, 'OverallStats_' +
                                                            str(num_images) + '_Images' + str(model_checkpoint)[:-4] +
                                                            '_Checkpoint_' + str(iou_score_threshold_test) + '_IoUPredThreshold_'+
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh' +
                                                            '_RunType_' + str(file_note)),
                                                           save_to_file=False)

                df_confusionmatrixdefectID = get_confusionmatrix_defectID(t_111_p_111=t_111_p_111,
                                                                          t_111_p_bdot=t_111_p_bdot,
                                                                          t_111_p_100=t_111_p_100,
                                                                          t_bdot_p_111=t_bdot_p_111,
                                                                          t_bdot_p_bdot=t_bdot_p_bdot,
                                                                          t_bdot_p_100=t_bdot_p_100,
                                                                          t_100_p_111=t_100_p_111,
                                                                          t_100_p_bdot=t_100_p_bdot,
                                                                          t_100_p_100=t_100_p_100,
                                                                          model_checkpoint=model_checkpoint,
                                                                          iou_score_threshold_test=iou_score_threshold_test,
                                                                          true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                         save_path=os.path.join(cfg.OUTPUT_DIR,
                                                         'ConfusionMatrixDefectID_' + str(num_images) + '_Images' + str(
                                                             model_checkpoint)[:-4] + '_Checkpoint_' + str(
                                                             iou_score_threshold_test) + '_IoUPredThreshold_'+
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh' + '_RunType_' + str(
                                                             file_note)),
                                                         save_to_file= False)

                df_computedstatsdefectID = get_computedstats_defectID(t_111_p_111=t_111_p_111,
                                                                    t_111_p_bdot=t_111_p_bdot,
                                                                    t_111_p_100=t_111_p_100,
                                                                    t_bdot_p_111=t_bdot_p_111,
                                                                    t_bdot_p_bdot=t_bdot_p_bdot,
                                                                    t_bdot_p_100=t_bdot_p_100,
                                                                    t_100_p_111=t_100_p_111,
                                                                    t_100_p_bdot=t_100_p_bdot,
                                                                    t_100_p_100=t_100_p_100,
                                                                      model_checkpoint=model_checkpoint,
                                                                      iou_score_threshold_test=iou_score_threshold_test,
                                                                      true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                    save_path=os.path.join(cfg.OUTPUT_DIR,
                                                                    'ComputedStatsDefectID_' + str(num_images) + '_Images' + str(
                                                                    model_checkpoint)[:-4] + '_Checkpoint_' + str(
                                                                    iou_score_threshold_test) + '_IoUPredThreshold_'+
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh' + '_RunType_' + str(
                                                                    file_note)),
                                                                    save_to_file=False)

                df_defectnumbers = get_defect_number_densities(true_classes_all_flattened=true_classes_all_flattened,
                                                               pred_classes_all_flattened=pred_classes_all_flattened,
                                                               num_images=num_images,
                                                               validation_image_filenames=validation_image_filenames,
                                                               model_checkpoint=model_checkpoint,
                                                            iou_score_threshold_test=iou_score_threshold_test,
                                                               true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                               save_path=os.path.join(cfg.OUTPUT_DIR,'DefectNumbers_' + str(
                                                                        num_images) + '_Images' + str(model_checkpoint)[:-4]
                                                                        + '_Checkpoint_' + str(iou_score_threshold_test) +
                                                                        '_IoUPredThreshold_' +
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh'+ '_RunType_' + str(file_note)),
                                                               save_to_file=False)

                df_defectsizes_FOUNDONLY = get_defect_sizes_average_and_errors(true_defectsizes_bdot_nm=true_defectsizes_bdot_nm_foundonly,
                                                                     true_defectsizes_111_nm=true_defectsizes_111_nm_foundonly,
                                                                     true_defectsizes_100_nm=true_defectsizes_100_nm_foundonly,
                                                                     pred_defectsizes_bdot_nm=pred_defectsizes_bdot_nm_foundonly,
                                                                     pred_defectsizes_111_nm=pred_defectsizes_111_nm_foundonly,
                                                                     pred_defectsizes_100_nm=pred_defectsizes_100_nm_foundonly,
                                                                    true_defectshapes_bdot=true_defectshapes_bdot_foundonly,
                                                                    true_defectshapes_111=true_defectshapes_111_foundonly,
                                                                    true_defectshapes_100=true_defectshapes_100_foundonly,
                                                                    pred_defectshapes_bdot=pred_defectshapes_bdot_foundonly,
                                                                    pred_defectshapes_111=pred_defectshapes_111_foundonly,
                                                                    pred_defectshapes_100=pred_defectshapes_100_foundonly,
                                                                     model_checkpoint=model_checkpoint,
                                                                     iou_score_threshold_test=iou_score_threshold_test,
                                                                     true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                    save_path=os.path.join(cfg.OUTPUT_DIR,
                                                                    'DefectSizes_FOUNDONLY' + str(num_images) + '_Images' + str(model_checkpoint)[:-4] +
                                                                    '_Checkpoint_' + str(iou_score_threshold_test) + '_IoUPredThreshold_' +
                                                                    str(true_and_pred_matching_threshold)+'_TruePredMatchThresh'+
                                                                    '_RunType_' + str(file_note)),
                                                                     save_to_file=False,
                                                                    cfg=cfg,
                                                                    file_string='FOUNDONLY')

                df_defectsizes_ALL = get_defect_sizes_average_and_errors(true_defectsizes_bdot_nm=true_defectsizes_bdot_nm_all,
                                                                     true_defectsizes_111_nm=true_defectsizes_111_nm_all,
                                                                     true_defectsizes_100_nm=true_defectsizes_100_nm_all,
                                                                     pred_defectsizes_bdot_nm=pred_defectsizes_bdot_nm_all,
                                                                     pred_defectsizes_111_nm=pred_defectsizes_111_nm_all,
                                                                     pred_defectsizes_100_nm=pred_defectsizes_100_nm_all,
                                                                         true_defectshapes_bdot=true_defectshapes_bdot_all,
                                                                         true_defectshapes_111=true_defectshapes_111_all,
                                                                         true_defectshapes_100=true_defectshapes_100_all,
                                                                         pred_defectshapes_bdot=pred_defectshapes_bdot_all,
                                                                         pred_defectshapes_111=pred_defectshapes_111_all,
                                                                         pred_defectshapes_100=pred_defectshapes_100_all,
                                                                     model_checkpoint=model_checkpoint,
                                                                     iou_score_threshold_test=iou_score_threshold_test,
                                                                     true_and_pred_matching_threshold=true_and_pred_matching_threshold,
                                                                    save_path=os.path.join(cfg.OUTPUT_DIR,
                                                                    'DefectSizes_ALL' + str(num_images) + '_Images' + str(model_checkpoint)[:-4] +
                                                                    '_Checkpoint_' + str(iou_score_threshold_test) + '_IoUPredThreshold_' +
                                                                    str(true_and_pred_matching_threshold)+'_TruePredMatchThresh'+
                                                                    '_RunType_' + str(file_note)),
                                                                     save_to_file=False,
                                                                    cfg=cfg,
                                                                    file_string='ALLDEFECTS')

                # Also make dataframes of defect size and density errors from per-image calculations
                data_dict_sizes = {'image names': image_name_list,
                            'bd size percent error per image': percent_size_error_bd_perimage_list,
                             '111 size percent error per image': percent_size_error_111_perimage_list,
                             '100 size percent error per image': percent_size_error_100_perimage_list,
                             'bd average size percent error per image': [np.mean(percent_size_error_bd_perimage_list)],
                             'bd stdev size percent error per image': [np.std(percent_size_error_bd_perimage_list)],
                             '111 average size percent error per image': [np.mean(percent_size_error_111_perimage_list)],
                             '111 stdev size percent error per image': [np.std(percent_size_error_111_perimage_list)],
                             '100 average size percent error per image': [np.mean(percent_size_error_100_perimage_list)],
                             '100 stdev size percent error per image': [np.std(percent_size_error_100_perimage_list)]
                             }
                df_defectsizes_perimage = pd.DataFrame.from_dict(data=data_dict_sizes, orient='index')

                data_dict_densities = {'image names': image_name_list,
                            'bd density percent error per image': percent_density_error_bd_perimage_list,
                             '111 density percent error per image': percent_density_error_111_perimage_list,
                             '100 density percent error per image': percent_density_error_100_perimage_list,
                             'bd average density percent error per image': [np.mean(percent_density_error_bd_perimage_list)],
                             'bd stdev density percent error per image': [np.std(percent_density_error_bd_perimage_list)],
                             '111 average density percent error per image': [np.mean(percent_density_error_111_perimage_list)],
                             '111 stdev density percent error per image': [np.std(percent_density_error_111_perimage_list)],
                             '100 average density percent error per image': [np.mean(percent_density_error_100_perimage_list)],
                             '100 stdev density percent error per image': [np.std(percent_density_error_100_perimage_list)]
                             }
                df_defectnumbers_perimage = pd.DataFrame.from_dict(data=data_dict_densities, orient='index')

                # Get classification report for pixel-level data
                #if input_yaml['mask_on'] == True:
                #    combined_report_asdict_pixels = get_pixel_classification_report(cfg=cfg, filename='CombinedReport_'+str(num_images)+'_Images'+str(model_checkpoint)[:-4]+'_Checkpoint_'+str(iou_score_threshold_test)+'_IoUThreshold'+'_RunType_'+str(file_note), true_pixels_all=true_pixels_all_flattened, pred_pixels_all=pred_pixels_all_flattened)
                #    classification_reports_all_checkpoints_pixels[str(model_checkpoint)[:-4]] = combined_report_asdict_pixels
                #else:
                #    classification_reports_all_checkpoints_pixels[str(model_checkpoint)[:-4]] = dict()


                # At end of this checkpoint analysis, write final report Excel file with all relevant dfs
                list_dfs = [df_overallstats, df_defectsizes_FOUNDONLY, df_defectsizes_ALL, df_defectsizes_perimage,
                            df_defectnumbers, df_defectnumbers_perimage,
                            df_confusionmatrixdefectID, df_computedstatsdefectID]

                dict_dfs[true_and_pred_matching_threshold] = list_dfs

                save_excel_together_singlereport(list_dfs=list_dfs,
                                                sheet_names=['OverallStats', 'DefectSizes_FoundOnly', 'DefectSizes_All',
                                                             'DefectSizes_PerImage',
                                                             'DefectNumbers', 'DefectNumbers_PerImage',
                                                             'ConfusionMatrixDefectID', 'ComputedStatsDefectID'],
                                                save_path=os.path.join(cfg.OUTPUT_DIR,
                                                                       'SingleReport_'+str(num_images)+'_Images_'+
                                                                       str(model_checkpoint)[:-4]+'_Checkpoint_'+str(iou_score_threshold_test)+
                                                                       '_IoUPredThreshold_'+
                                                                       str(true_and_pred_matching_threshold)+
                                                                       '_TruePredMatchThresh'+'_RunType_'+str(file_note)+'.xlsx'))

                ##########
                #
                # Here- make parity plots of true vs. pred avg defect sizes
                #
                ##########
                # black dot sizes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_size_bd_perimage_list = list(np.array(true_avg_size_bd_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_bd_perimage_list)))])
                pred_avg_size_bd_perimage_list = list(np.array(pred_avg_size_bd_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_bd_perimage_list)))])
                true_avg_size_bd_perimage_list = list(np.array(true_avg_size_bd_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_bd_perimage_list)))])
                pred_avg_size_bd_perimage_list = list(np.array(pred_avg_size_bd_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_bd_perimage_list)))])
                image_name_list_bd_size = copy(image_name_list)
                image_name_list_bd_size = list(np.array(image_name_list_bd_size)[np.where(~np.isnan(np.array(true_avg_size_bd_perimage_list)))])
                image_name_list_bd_size = list(np.array(image_name_list_bd_size)[np.where(~np.isnan(np.array(pred_avg_size_bd_perimage_list)))])
                ax.scatter(true_avg_size_bd_perimage_list, pred_avg_size_bd_perimage_list, color='blue', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg black dot sizes per image (nm)', fontsize=12)
                ax.set_ylabel('Predicted avg black dot sizes per image (nm)', fontsize=12)
                xlow = int(min(true_avg_size_bd_perimage_list) - 0.1*(max(true_avg_size_bd_perimage_list)-min(true_avg_size_bd_perimage_list)))
                xhigh = int(max(true_avg_size_bd_perimage_list) + 0.1*(max(true_avg_size_bd_perimage_list)-min(true_avg_size_bd_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_size_bd_perimage_list, pred_avg_size_bd_perimage_list)
                mae = mean_absolute_error(true_avg_size_bd_perimage_list, pred_avg_size_bd_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_size_bd_perimage_list, pred_avg_size_bd_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectSize_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Blackdot' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_bd_size
                data_dict['true bd size'] = true_avg_size_bd_perimage_list
                data_dict['pred bd size'] = pred_avg_size_bd_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectSize_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Blackdot' + '.xlsx'))


                # 111 loop sizes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_size_111_perimage_list = list(np.array(true_avg_size_111_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_111_perimage_list)))])
                pred_avg_size_111_perimage_list = list(np.array(pred_avg_size_111_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_111_perimage_list)))])
                true_avg_size_111_perimage_list = list(np.array(true_avg_size_111_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_111_perimage_list)))])
                pred_avg_size_111_perimage_list = list(np.array(pred_avg_size_111_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_111_perimage_list)))])
                image_name_list_111_size = copy(image_name_list)
                image_name_list_111_size = list(np.array(image_name_list_111_size)[np.where(~np.isnan(np.array(true_avg_size_111_perimage_list)))])
                image_name_list_111_size = list(np.array(image_name_list_111_size)[np.where(~np.isnan(np.array(pred_avg_size_111_perimage_list)))])
                ax.scatter(true_avg_size_111_perimage_list, pred_avg_size_111_perimage_list, color='red', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg <111> sizes per image (nm)', fontsize=12)
                ax.set_ylabel('Predicted avg <111> sizes per image (nm)', fontsize=12)
                xlow = int(min(true_avg_size_111_perimage_list) - 0.1*(max(true_avg_size_111_perimage_list)-min(true_avg_size_111_perimage_list)))
                xhigh = int(max(true_avg_size_111_perimage_list) + 0.1*(max(true_avg_size_111_perimage_list)-min(true_avg_size_111_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_size_111_perimage_list, pred_avg_size_111_perimage_list)
                mae = mean_absolute_error(true_avg_size_111_perimage_list, pred_avg_size_111_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_size_111_perimage_list, pred_avg_size_111_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'root_mean_squared_error': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectSize_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_111Loop' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_111_size
                data_dict['true 111 size'] = true_avg_size_111_perimage_list
                data_dict['pred 111 size'] = pred_avg_size_111_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectSize_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_111Loop' + '.xlsx'))

                # 100 loop sizes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_size_100_perimage_list = list(np.array(true_avg_size_100_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_100_perimage_list)))])
                pred_avg_size_100_perimage_list = list(np.array(pred_avg_size_100_perimage_list)[np.where(~np.isnan(np.array(pred_avg_size_100_perimage_list)))])
                true_avg_size_100_perimage_list = list(np.array(true_avg_size_100_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_100_perimage_list)))])
                pred_avg_size_100_perimage_list = list(np.array(pred_avg_size_100_perimage_list)[np.where(~np.isnan(np.array(true_avg_size_100_perimage_list)))])
                image_name_list_100_size = copy(image_name_list)
                image_name_list_100_size = list(np.array(image_name_list_100_size)[np.where(~np.isnan(np.array(true_avg_size_100_perimage_list)))])
                image_name_list_100_size = list(np.array(image_name_list_100_size)[np.where(~np.isnan(np.array(pred_avg_size_100_perimage_list)))])
                ax.scatter(true_avg_size_100_perimage_list, pred_avg_size_100_perimage_list, color='yellow',
                           edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg <100> sizes per image (nm)', fontsize=12)
                ax.set_ylabel('Predicted avg <100> sizes per image (nm)', fontsize=12)
                xlow = int(min(true_avg_size_100_perimage_list) - 0.1*(max(true_avg_size_100_perimage_list)-min(true_avg_size_100_perimage_list)))
                xhigh = int(max(true_avg_size_100_perimage_list) + 0.1*(max(true_avg_size_100_perimage_list)-min(true_avg_size_100_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_size_100_perimage_list, pred_avg_size_100_perimage_list)
                mae = mean_absolute_error(true_avg_size_100_perimage_list, pred_avg_size_100_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_size_100_perimage_list, pred_avg_size_100_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'root_mean_squared_error': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectSize_ParityPlot_TruePredMatch_' + str(
                    true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test) + '_100Loop' + '.png'),
                            dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_100_size
                data_dict['true 100 size'] = true_avg_size_100_perimage_list
                data_dict['pred 100 size'] = pred_avg_size_100_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectSize_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_100Loop' + '.xlsx'))


                # All defect sizes together
                all_avg_true_sizes = true_avg_size_100_perimage_list + true_avg_size_111_perimage_list + true_avg_size_bd_perimage_list
                all_avg_pred_sizes = pred_avg_size_100_perimage_list + pred_avg_size_111_perimage_list + pred_avg_size_bd_perimage_list
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                ax.scatter(true_avg_size_bd_perimage_list, pred_avg_size_bd_perimage_list, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='black dot')
                ax.scatter(true_avg_size_111_perimage_list, pred_avg_size_111_perimage_list, color='red',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<111> loop')
                ax.scatter(true_avg_size_100_perimage_list, pred_avg_size_100_perimage_list, color='yellow',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<100> loop')
                ax.legend(loc='lower right')
                #ax.scatter(all_avg_true_sizes, all_avg_pred_sizes, color='black',
                #           edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg defect sizes per image (nm)', fontsize=12)
                ax.set_ylabel('Predicted avg defect sizes per image (nm)', fontsize=12)
                xlow = int(min(all_avg_true_sizes) - 0.1*(max(all_avg_true_sizes)-min(all_avg_true_sizes)))
                xhigh = int(max(all_avg_true_sizes) + 0.1*(max(all_avg_true_sizes)-min(all_avg_true_sizes)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(all_avg_true_sizes, all_avg_pred_sizes)
                mae = mean_absolute_error(all_avg_true_sizes, all_avg_pred_sizes)
                rmse = np.sqrt(mean_squared_error(all_avg_true_sizes, all_avg_pred_sizes))
                d = {'R2': r2, 'MAE': mae, 'root_mean_squared_error': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectSize_ParityPlot_TruePredMatch_' + str(
                    true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'),
                            dpi=250, bbox_inches='tight')

                # All defect sizes avg and stdev parity plot
                true_sizes_100_avg = np.mean(true_avg_size_100_perimage_list)
                true_sizes_100_std = np.std(true_avg_size_100_perimage_list)
                pred_sizes_100_avg = np.mean(pred_avg_size_100_perimage_list)
                pred_sizes_100_std = np.std(pred_avg_size_100_perimage_list)
                true_sizes_111_avg = np.mean(true_avg_size_111_perimage_list)
                true_sizes_111_std = np.std(true_avg_size_111_perimage_list)
                pred_sizes_111_avg = np.mean(pred_avg_size_111_perimage_list)
                pred_sizes_111_std = np.std(pred_avg_size_111_perimage_list)
                true_sizes_bd_avg = np.mean(true_avg_size_bd_perimage_list)
                true_sizes_bd_std = np.std(true_avg_size_bd_perimage_list)
                pred_sizes_bd_avg = np.mean(pred_avg_size_bd_perimage_list)
                pred_sizes_bd_std = np.std(pred_avg_size_bd_perimage_list)
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                ax.scatter(true_sizes_bd_avg, pred_sizes_bd_avg, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='black dot average')
                #yerr = np.array([pred_sizes_bd_std, pred_sizes_bd_std])
                #xerr = np.array([true_sizes_bd_std, true_sizes_bd_std])
                ax.errorbar(true_sizes_bd_avg, pred_sizes_bd_avg, xerr=true_sizes_bd_std, yerr=pred_sizes_bd_std, capsize=2, ecolor='k', linestyle='none', label=None)

                ax.scatter(true_sizes_111_avg, pred_sizes_111_avg, color='red',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<111> loop average')
                #yerr = np.array([pred_sizes_111_std, pred_sizes_111_std])
                #xerr = np.array([true_sizes_111_std, true_sizes_111_std])
                ax.errorbar(true_sizes_111_avg, pred_sizes_111_avg, xerr=true_sizes_111_std, yerr=pred_sizes_111_std, capsize=2, ecolor='k', linestyle='none', label=None)

                ax.scatter(true_sizes_100_avg, pred_sizes_100_avg, color='yellow',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<100> loop average')
                #yerr = np.array([pred_sizes_100_std, pred_sizes_100_std])
                #xerr = np.array([true_sizes_100_std, true_sizes_100_std])
                ax.errorbar(true_sizes_100_avg, pred_sizes_100_avg, xerr=true_sizes_100_std, yerr=pred_sizes_100_std, capsize=2, ecolor='k', linestyle='none', label=None)
                ax.legend(loc='lower right')
                ax.set_xlabel('True average defect sizes (nm)', fontsize=12)
                ax.set_ylabel('Predicted average defect sizes (nm)', fontsize=12)
                xlow = int(min(all_avg_true_sizes) - 0.1*(max(all_avg_true_sizes)-min(all_avg_true_sizes)))
                xhigh = int(max(all_avg_true_sizes) + 0.1*(max(all_avg_true_sizes)-min(all_avg_true_sizes)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score([true_sizes_111_avg, true_sizes_100_avg, true_sizes_bd_avg], [pred_sizes_111_avg, pred_sizes_100_avg, pred_sizes_bd_avg])
                mae = mean_absolute_error([true_sizes_111_avg, true_sizes_100_avg, true_sizes_bd_avg], [pred_sizes_111_avg, pred_sizes_100_avg, pred_sizes_bd_avg])
                rmse = np.sqrt(mean_squared_error([true_sizes_111_avg, true_sizes_100_avg, true_sizes_bd_avg], [pred_sizes_111_avg, pred_sizes_100_avg, pred_sizes_bd_avg]))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectSize_AvgStdev_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'), dpi=250, bbox_inches='tight')


                ##########
                #
                # Here- make parity plots of true vs. pred  defect densities
                #
                ##########
                # black dot densities
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                #true_density_bd_perimage_list = list(10**4*np.array(true_density_bd_perimage_list)[np.where(~np.isnan(np.array(pred_density_bd_perimage_list)))])
                #pred_density_bd_perimage_list = list(10**4*np.array(pred_density_bd_perimage_list)[np.where(~np.isnan(np.array(pred_density_bd_perimage_list)))])
                #true_density_bd_perimage_list = list(10**4*np.array(true_density_bd_perimage_list)[np.where(~np.isnan(np.array(true_density_bd_perimage_list)))])
                #pred_density_bd_perimage_list = list(10**4*np.array(pred_density_bd_perimage_list)[np.where(~np.isnan(np.array(true_density_bd_perimage_list)))])
                true_density_bd_perimage_list = list(10**4*np.array(true_density_bd_perimage_list)[np.where(~np.isnan(np.array(pred_density_bd_perimage_list)) & ~np.isnan(np.array(true_density_bd_perimage_list)))])
                pred_density_bd_perimage_list = list(10**4*np.array(pred_density_bd_perimage_list)[np.where(~np.isnan(np.array(pred_density_bd_perimage_list)) & ~np.isnan(np.array(true_density_bd_perimage_list)))])
                image_name_list_bd_density = copy(image_name_list)
                image_name_list_bd_density = list(np.array(image_name_list_bd_density)[np.where(~np.isnan(np.array(true_density_bd_perimage_list)))])
                image_name_list_bd_density = list(np.array(image_name_list_bd_density)[np.where(~np.isnan(np.array(pred_density_bd_perimage_list)))])
                ax.scatter(true_density_bd_perimage_list, pred_density_bd_perimage_list, color='blue', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True black dot densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                ax.set_ylabel('Predicted black dot densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                xlow = int(min(true_density_bd_perimage_list) - 0.1*(max(true_density_bd_perimage_list)-min(true_density_bd_perimage_list)))
                xhigh = int(max(true_density_bd_perimage_list) + 0.1*(max(true_density_bd_perimage_list)-min(true_density_bd_perimage_list)))
                #xlow = 0.0
                #xhigh = 10.0
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_density_bd_perimage_list, pred_density_bd_perimage_list)
                mae = mean_absolute_error(true_density_bd_perimage_list, pred_density_bd_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_density_bd_perimage_list, pred_density_bd_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Blackdot' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_bd_density
                data_dict['true bd density'] = true_density_bd_perimage_list
                data_dict['pred bd density'] = pred_density_bd_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Blackdot' + '.xlsx'))

                # 111 loop densities
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                #true_density_111_perimage_list = list(10**4*np.array(true_density_111_perimage_list)[np.where(~np.isnan(np.array(pred_density_111_perimage_list)))])
                #pred_density_111_perimage_list = list(10**4*np.array(pred_density_111_perimage_list)[np.where(~np.isnan(np.array(pred_density_111_perimage_list)))])
                #true_density_111_perimage_list = list(10**4*np.array(true_density_111_perimage_list)[np.where(~np.isnan(np.array(true_density_111_perimage_list)))])
                #pred_density_111_perimage_list = list(10**4*np.array(pred_density_111_perimage_list)[np.where(~np.isnan(np.array(true_density_111_perimage_list)))])
                true_density_111_perimage_list = list(10**4*np.array(true_density_111_perimage_list)[np.where(~np.isnan(np.array(pred_density_111_perimage_list)) & ~np.isnan(np.array(true_density_111_perimage_list)))])
                pred_density_111_perimage_list = list(10**4*np.array(pred_density_111_perimage_list)[np.where(~np.isnan(np.array(pred_density_111_perimage_list)) & ~np.isnan(np.array(true_density_111_perimage_list)))])
                image_name_list_111_density = copy(image_name_list)
                image_name_list_111_density = list(np.array(image_name_list_111_density)[np.where(~np.isnan(np.array(true_density_111_perimage_list)))])
                image_name_list_111_density = list(np.array(image_name_list_111_density)[np.where(~np.isnan(np.array(pred_density_111_perimage_list)))])
                ax.scatter(true_density_111_perimage_list, pred_density_111_perimage_list, color='red', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True <111> densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                ax.set_ylabel('Predicted <111> densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                xlow = int(min(true_density_111_perimage_list) - 0.1*(max(true_density_111_perimage_list)-min(true_density_111_perimage_list)))
                xhigh = int(max(true_density_111_perimage_list) + 0.1*(max(true_density_111_perimage_list)-min(true_density_111_perimage_list)))
                #xlow = 0.0
                #xhigh = 10.0
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_density_111_perimage_list, pred_density_111_perimage_list)
                mae = mean_absolute_error(true_density_111_perimage_list, pred_density_111_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_density_111_perimage_list, pred_density_111_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_111Loop' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_111_density
                data_dict['true 111 density'] = true_density_111_perimage_list
                data_dict['pred 111 density'] = pred_density_111_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_111Loop' + '.xlsx'))

                # 100 loop densities
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                #true_density_100_perimage_list = list(10**4*np.array(true_density_100_perimage_list)[np.where(~np.isnan(np.array(pred_density_100_perimage_list)))])
                #pred_density_100_perimage_list = list(10**4*np.array(pred_density_100_perimage_list)[np.where(~np.isnan(np.array(pred_density_100_perimage_list)))])
                #true_density_100_perimage_list = list(10 ** 4 * np.array(true_density_100_perimage_list)[np.where(~np.isnan(np.array(true_density_100_perimage_list)))])
                #pred_density_100_perimage_list = list(10 ** 4 * np.array(pred_density_100_perimage_list)[np.where(~np.isnan(np.array(true_density_100_perimage_list)))])
                true_density_100_perimage_list = list(10**4*np.array(true_density_100_perimage_list)[np.where(~np.isnan(np.array(pred_density_100_perimage_list)) & ~np.isnan(np.array(true_density_100_perimage_list)))])
                pred_density_100_perimage_list = list(10**4*np.array(pred_density_100_perimage_list)[np.where(~np.isnan(np.array(pred_density_100_perimage_list)) & ~np.isnan(np.array(true_density_100_perimage_list)))])
                image_name_list_100_density = copy(image_name_list)
                image_name_list_100_density = list(np.array(image_name_list_100_density)[np.where(~np.isnan(np.array(true_density_100_perimage_list)))])
                image_name_list_100_density = list(np.array(image_name_list_100_density)[np.where(~np.isnan(np.array(pred_density_100_perimage_list)))])
                ax.scatter(true_density_100_perimage_list, pred_density_100_perimage_list, color='yellow', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True <100> densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                ax.set_ylabel('Predicted <100> densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                xlow = int(min(true_density_100_perimage_list) - 0.1*(max(true_density_100_perimage_list)-min(true_density_100_perimage_list)))
                xhigh = int(max(true_density_100_perimage_list) + 0.1*(max(true_density_100_perimage_list)-min(true_density_100_perimage_list)))
                #xlow = 0.0
                #xhigh = 10.0
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_density_100_perimage_list, pred_density_100_perimage_list)
                mae = mean_absolute_error(true_density_100_perimage_list, pred_density_100_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_density_100_perimage_list, pred_density_100_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_100Loop' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_100_density
                data_dict['true 100 density'] = true_density_100_perimage_list
                data_dict['pred 100 density'] = pred_density_100_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_100Loop' + '.xlsx'))

                # All defect densities together
                all_true_densities = true_density_100_perimage_list+true_density_111_perimage_list+true_density_bd_perimage_list
                all_pred_densities = pred_density_100_perimage_list+pred_density_111_perimage_list+pred_density_bd_perimage_list
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                #ax.scatter(all_true_densities, all_pred_densities, color='black', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.scatter(true_density_bd_perimage_list, pred_density_bd_perimage_list, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='black dot')
                ax.scatter(true_density_111_perimage_list, pred_density_111_perimage_list, color='red',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<111> loop')
                ax.scatter(true_density_100_perimage_list, pred_density_100_perimage_list, color='yellow',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<100> loop')
                ax.legend(loc='lower right')
                ax.set_xlabel('True defect densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                ax.set_ylabel('Predicted defect densities per image (x10$^4$ #/nm$^2$)', fontsize=12)
                xlow = int(min(all_true_densities) - 0.1*(max(all_true_densities)-min(all_true_densities)))
                xhigh = int(max(all_true_densities) + 0.1*(max(all_true_densities)-min(all_true_densities)))
                #xlow = 0.0
                #xhigh = 10.0
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(all_true_densities, all_pred_densities)
                mae = mean_absolute_error(all_true_densities, all_pred_densities)
                rmse = np.sqrt(mean_squared_error(all_true_densities, all_pred_densities))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectDensity_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'), dpi=250, bbox_inches='tight')

                # All defect densities avg and stdev parity plot
                true_density_100_avg = np.mean(true_density_100_perimage_list)
                true_density_100_std = np.std(true_density_100_perimage_list)
                pred_density_100_avg = np.mean(pred_density_100_perimage_list)
                pred_density_100_std = np.std(pred_density_100_perimage_list)
                true_density_111_avg = np.mean(true_density_111_perimage_list)
                true_density_111_std = np.std(true_density_111_perimage_list)
                pred_density_111_avg = np.mean(pred_density_111_perimage_list)
                pred_density_111_std = np.std(pred_density_111_perimage_list)
                true_density_bd_avg = np.mean(true_density_bd_perimage_list)
                true_density_bd_std = np.std(true_density_bd_perimage_list)
                pred_density_bd_avg = np.mean(pred_density_bd_perimage_list)
                pred_density_bd_std = np.std(pred_density_bd_perimage_list)
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                ax.scatter(true_density_bd_avg, pred_density_bd_avg, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='black dot average')
                #yerr = np.array([pred_density_bd_std, pred_density_bd_std])
                #xerr = np.array([true_density_bd_std, true_density_bd_std])
                ax.errorbar(true_density_bd_avg, pred_density_bd_avg, xerr=true_density_bd_std, yerr=pred_density_bd_std, capsize=2, ecolor='k', linestyle='none', label=None)

                ax.scatter(true_density_111_avg, pred_density_111_avg, color='red',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<111> loop average')
                #yerr = np.array([pred_density_111_std, pred_density_111_std])
                #xerr = np.array([true_density_111_std, true_density_111_std])
                ax.errorbar(true_density_111_avg, pred_density_111_avg, xerr=true_density_111_std, yerr=pred_density_111_std, capsize=2, ecolor='k', linestyle='none', label=None)

                ax.scatter(true_density_100_avg, pred_density_100_avg, color='yellow',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<100> loop average')
                #yerr = np.array([pred_density_100_std, pred_density_100_std])
                #xerr = np.array([true_density_100_std, true_density_100_std])
                ax.errorbar(true_density_100_avg, pred_density_100_avg, xerr=true_density_100_std, yerr=pred_density_100_std, capsize=2, ecolor='k', linestyle='none', label=None)
                ax.legend(loc='lower right')
                ax.set_xlabel('True average defect densities (x10$^4$ #/nm$^2$)', fontsize=12)
                ax.set_ylabel('Predicted average defect densities (x10$^4$ #/nm$^2$)', fontsize=12)
                xlow = 0.0
                xhigh = 5.0
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score([true_density_111_avg, true_density_100_avg, true_density_bd_avg], [pred_density_111_avg, pred_density_100_avg, pred_density_bd_avg])
                mae = mean_absolute_error([true_density_111_avg, true_density_100_avg, true_density_bd_avg], [pred_density_111_avg, pred_density_100_avg, pred_density_bd_avg])
                rmse = np.sqrt(mean_squared_error([true_density_111_avg, true_density_100_avg, true_density_bd_avg], [pred_density_111_avg, pred_density_100_avg, pred_density_bd_avg]))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectDensity_AvgStdev_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'), dpi=250, bbox_inches='tight')

                ##########
                #
                # Here- make parity plots of true vs. pred avg defect shapes
                #
                ##########
                # black dot shapes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_shape_bd_perimage_list = list(np.array(true_avg_shape_bd_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_bd_perimage_list)))])
                pred_avg_shape_bd_perimage_list = list(np.array(pred_avg_shape_bd_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_bd_perimage_list)))])
                true_avg_shape_bd_perimage_list = list(np.array(true_avg_shape_bd_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_bd_perimage_list)))])
                pred_avg_shape_bd_perimage_list = list(np.array(pred_avg_shape_bd_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_bd_perimage_list)))])
                image_name_list_bd_shape = copy(image_name_list)
                image_name_list_bd_shape = list(np.array(image_name_list_bd_shape)[np.where(~np.isnan(np.array(true_avg_shape_bd_perimage_list)))])
                image_name_list_bd_shape = list(np.array(image_name_list_bd_shape)[np.where(~np.isnan(np.array(pred_avg_shape_bd_perimage_list)))])
                ax.scatter(true_avg_shape_bd_perimage_list, pred_avg_shape_bd_perimage_list, color='blue', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg black dot Heywood circularity per image', fontsize=12)
                ax.set_ylabel('Predicted avg black dot Heywood circularity per image', fontsize=12)
                xlow = 0.9
                xhigh = 2.5
                #xlow = int(min(true_avg_shape_bd_perimage_list) - 0.1*(max(true_avg_shape_bd_perimage_list)-min(true_avg_shape_bd_perimage_list)))
                #xhigh = int(max(true_avg_shape_bd_perimage_list) + 0.1*(max(true_avg_shape_bd_perimage_list)-min(true_avg_shape_bd_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_shape_bd_perimage_list, pred_avg_shape_bd_perimage_list)
                mae = mean_absolute_error(true_avg_shape_bd_perimage_list, pred_avg_shape_bd_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_shape_bd_perimage_list, pred_avg_shape_bd_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Blackdot' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_bd_shape
                data_dict['true bd shape'] = true_avg_shape_bd_perimage_list
                data_dict['pred bd shape'] = pred_avg_shape_bd_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_Blackdot' + '.xlsx'))

                # 111 shapes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_shape_111_perimage_list = list(np.array(true_avg_shape_111_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_111_perimage_list)))])
                pred_avg_shape_111_perimage_list = list(np.array(pred_avg_shape_111_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_111_perimage_list)))])
                true_avg_shape_111_perimage_list = list(np.array(true_avg_shape_111_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_111_perimage_list)))])
                pred_avg_shape_111_perimage_list = list(np.array(pred_avg_shape_111_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_111_perimage_list)))])
                image_name_list_111_shape = copy(image_name_list)
                image_name_list_111_shape = list(np.array(image_name_list_111_shape)[np.where(~np.isnan(np.array(true_avg_shape_111_perimage_list)))])
                image_name_list_111_shape = list(np.array(image_name_list_111_shape)[np.where(~np.isnan(np.array(pred_avg_shape_111_perimage_list)))])
                ax.scatter(true_avg_shape_111_perimage_list, pred_avg_shape_111_perimage_list, color='red', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg <111> loop Heywood circularity per image', fontsize=12)
                ax.set_ylabel('Predicted avg <111> loop Heywood circularity per image', fontsize=12)
                xlow = 0.9
                xhigh = 2.5
                #xlow = int(min(true_avg_shape_111_perimage_list) - 0.1*(max(true_avg_shape_111_perimage_list)-min(true_avg_shape_111_perimage_list)))
                #xhigh = int(max(true_avg_shape_111_perimage_list) + 0.1*(max(true_avg_shape_111_perimage_list)-min(true_avg_shape_111_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_shape_111_perimage_list, pred_avg_shape_111_perimage_list)
                mae = mean_absolute_error(true_avg_shape_111_perimage_list, pred_avg_shape_111_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_shape_111_perimage_list, pred_avg_shape_111_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_111Loop' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_111_shape
                data_dict['true 111 shape'] = true_avg_shape_111_perimage_list
                data_dict['pred 111 shape'] = pred_avg_shape_111_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_111Loop' + '.xlsx'))

                # 100 shapes
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                true_avg_shape_100_perimage_list = list(np.array(true_avg_shape_100_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_100_perimage_list)))])
                pred_avg_shape_100_perimage_list = list(np.array(pred_avg_shape_100_perimage_list)[np.where(~np.isnan(np.array(pred_avg_shape_100_perimage_list)))])
                true_avg_shape_100_perimage_list = list(np.array(true_avg_shape_100_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_100_perimage_list)))])
                pred_avg_shape_100_perimage_list = list(np.array(pred_avg_shape_100_perimage_list)[np.where(~np.isnan(np.array(true_avg_shape_100_perimage_list)))])
                image_name_list_100_shape = copy(image_name_list)
                image_name_list_100_shape = list(np.array(image_name_list_100_shape)[np.where(~np.isnan(np.array(true_avg_shape_100_perimage_list)))])
                image_name_list_100_shape = list(np.array(image_name_list_100_shape)[np.where(~np.isnan(np.array(pred_avg_shape_100_perimage_list)))])
                ax.scatter(true_avg_shape_100_perimage_list, pred_avg_shape_100_perimage_list, color='yellow', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.set_xlabel('True avg <100> loop Heywood circularity per image', fontsize=12)
                ax.set_ylabel('Predicted avg <100> loop Heywood circularity per image', fontsize=12)
                xlow = 0.9
                xhigh = 2.5
                #xlow = int(min(true_avg_shape_100_perimage_list) - 0.1*(max(true_avg_shape_100_perimage_list)-min(true_avg_shape_100_perimage_list)))
                #xhigh = int(max(true_avg_shape_100_perimage_list) + 0.1*(max(true_avg_shape_100_perimage_list)-min(true_avg_shape_100_perimage_list)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(true_avg_shape_100_perimage_list, pred_avg_shape_100_perimage_list)
                mae = mean_absolute_error(true_avg_shape_100_perimage_list, pred_avg_shape_100_perimage_list)
                rmse = np.sqrt(mean_squared_error(true_avg_shape_100_perimage_list, pred_avg_shape_100_perimage_list))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_100Loop' + '.png'), dpi=250, bbox_inches='tight')
                data_dict = dict()
                data_dict['image name'] = image_name_list_100_shape
                data_dict['true 100 shape'] = true_avg_shape_100_perimage_list
                data_dict['pred 100 shape'] = pred_avg_shape_100_perimage_list
                pd.DataFrame().from_dict(data=data_dict).to_excel(os.path.join(cfg.OUTPUT_DIR,'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_100Loop' + '.xlsx'))

                # All defect shapes together
                all_true_avg_shapes = true_avg_shape_100_perimage_list + true_avg_shape_111_perimage_list + true_avg_shape_bd_perimage_list
                all_pred_avg_shapes = pred_avg_shape_100_perimage_list + pred_avg_shape_111_perimage_list + pred_avg_shape_bd_perimage_list
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                #ax.scatter(all_true_avg_shapes, all_pred_avg_shapes, color='black', edgecolors='black', s=100, zorder=2, alpha=0.7)
                ax.scatter(true_avg_shape_bd_perimage_list, pred_avg_shape_bd_perimage_list, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label = 'black dot')
                ax.scatter(true_avg_shape_111_perimage_list, pred_avg_shape_111_perimage_list, color='red',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label = '<111> loop')
                ax.scatter(true_avg_shape_100_perimage_list, pred_avg_shape_100_perimage_list, color='yellow',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label = '<100> loop')
                ax.legend(loc='lower right')
                ax.set_xlabel('True avg defect Heywood circularity per image', fontsize=12)
                ax.set_ylabel('Predicted avg defect Heywood circularity per image', fontsize=12)
                xlow = 0.9
                xhigh = 2.5
                #xlow = int(min(all_true_avg_shapes) - 0.1*(max(all_true_avg_shapes)-min(all_true_avg_shapes)))
                #xhigh = int(max(all_true_avg_shapes) + 0.1*(max(all_true_avg_shapes)-min(all_true_avg_shapes)))
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score(all_true_avg_shapes, all_pred_avg_shapes)
                mae = mean_absolute_error(all_true_avg_shapes, all_pred_avg_shapes)
                rmse = np.sqrt(mean_squared_error(all_true_avg_shapes, all_pred_avg_shapes))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectShape_ParityPlot_TruePredMatch_' + str(true_and_pred_matching_threshold) +
                                 '_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'), dpi=250, bbox_inches='tight')

                # All defect shapes avg and stdev parity plot
                true_shape_100_avg = np.mean(true_avg_shape_100_perimage_list)
                true_shape_100_std = np.std(true_avg_shape_100_perimage_list)
                pred_shape_100_avg = np.mean(pred_avg_shape_100_perimage_list)
                pred_shape_100_std = np.std(pred_avg_shape_100_perimage_list)
                true_shape_111_avg = np.mean(true_avg_shape_111_perimage_list)
                true_shape_111_std = np.std(true_avg_shape_111_perimage_list)
                pred_shape_111_avg = np.mean(pred_avg_shape_111_perimage_list)
                pred_shape_111_std = np.std(pred_avg_shape_111_perimage_list)
                true_shape_bd_avg = np.mean(true_avg_shape_bd_perimage_list)
                true_shape_bd_std = np.std(true_avg_shape_bd_perimage_list)
                pred_shape_bd_avg = np.mean(pred_avg_shape_bd_perimage_list)
                pred_shape_bd_std = np.std(pred_avg_shape_bd_perimage_list)
                fig, ax = get_fig_ax(aspect_ratio=0.5, x_align=0.65)
                ax.scatter(true_shape_bd_avg, pred_shape_bd_avg, color='blue',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='black dot average')
                #yerr = np.array([pred_shape_bd_std, pred_shape_bd_std])
                #xerr = np.array([true_shape_bd_std, true_shape_bd_std])
                ax.errorbar(true_shape_bd_avg, pred_shape_bd_avg, xerr=true_shape_bd_std, yerr=pred_shape_bd_std, capsize=2, ecolor='k',
                            linestyle='none', label=None)

                ax.scatter(true_shape_111_avg, pred_shape_111_avg, color='red',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<111> loop average')
                #yerr = np.array([pred_shape_111_std, pred_shape_111_std])
                #xerr = np.array([true_shape_111_std, true_shape_111_std])
                ax.errorbar(true_shape_111_avg, pred_shape_111_avg, xerr=true_shape_111_std, yerr=pred_shape_111_std, capsize=2, ecolor='k',
                            linestyle='none', label=None)

                ax.scatter(true_shape_100_avg, pred_shape_100_avg, color='yellow',
                           edgecolors='black', s=100, zorder=2, alpha=0.7, label='<100> loop average')
                #yerr = np.array([pred_shape_100_std, pred_shape_100_std])
                #xerr = np.array([true_shape_100_std, true_shape_100_std])
                ax.errorbar(true_shape_100_avg, pred_shape_100_avg, xerr=true_shape_100_std, yerr=pred_shape_100_std, capsize=2, ecolor='k',
                            linestyle='none', label=None)
                ax.legend(loc='lower right')
                ax.set_xlabel('True average defect Heywood circularity', fontsize=12)
                ax.set_ylabel('Predicted average defect Heywood circularity', fontsize=12)
                xlow = 0.9
                xhigh = 2.0
                ax.set_xlim(left=xlow, right=xhigh)
                ax.set_ylim(bottom=xlow, top=xhigh)
                ax.plot([xlow, xhigh], [xlow, xhigh], color='black', lw=1.5, linestyle='--')
                r2 = r2_score([true_shape_100_avg, true_shape_111_avg, true_shape_bd_avg], [pred_shape_100_avg, pred_shape_111_avg, pred_shape_bd_avg])
                mae = mean_absolute_error([true_shape_100_avg, true_shape_111_avg, true_shape_bd_avg], [pred_shape_100_avg, pred_shape_111_avg, pred_shape_bd_avg])
                rmse = np.sqrt(mean_squared_error([true_shape_100_avg, true_shape_111_avg, true_shape_bd_avg], [pred_shape_100_avg, pred_shape_111_avg, pred_shape_bd_avg]))
                d = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                plot_stats(fig, d, x_align=0.15, y_align=0.90, type='float')
                fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'DefectShape_AvgStdev_ParityPlot_TruePredMatch_' + str(
                    true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test) + '_AllDefects' + '.png'),
                            dpi=250, bbox_inches='tight')

                # Output all key per-image stats to json file
                with open(os.path.join(cfg.OUTPUT_DIR, 'StatsPerImage_TruePredMatch_' + str(true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test)+'.json'), 'w') as f:
                    json.dump(data_dict_per_image, f)

                # Get best and worst image for each statistic, output to file
                best_f1 = 0
                best_find_f1 = 0
                best_density = 10**5
                best_size = 10**5
                data_dict_best_images = dict()
                for img, data in data_dict_per_image.items():
                    if data['avg size error'] < best_size:
                        best_size = data['avg size error']
                        data_dict_best_images['best avg size error'] = img
                    if data['avg density error'] < best_density:
                        best_density = data['avg density error']
                        data_dict_best_images['best avg density error'] = img
                    if data['overall F1'] > best_f1:
                        best_f1 = data['overall F1']
                        data_dict_best_images['best overall F1'] = img
                    if data['defect find F1'] > best_find_f1:
                        best_find_f1 = data['defect find F1']
                        data_dict_best_images['best defect find F1'] = img

                with open(os.path.join(cfg.OUTPUT_DIR, 'BestImagesPerStat_TruePredMatch_' + str(true_and_pred_matching_threshold) +'_IoUScoreThresh_' + str(iou_score_threshold_test)+'.json'), 'w') as f:
                    json.dump(data_dict_best_images, f)

                worst_f1 = 1
                worst_find_f1 = 1
                worst_density = 0
                worst_size = 0
                data_dict_worst_images = dict()
                for img, data in data_dict_per_image.items():
                    if data['avg size error'] > worst_size:
                        worst_size = data['avg size error']
                        data_dict_worst_images['worst avg size error'] = img
                    if data['avg density error'] > worst_density:
                        worst_density = data['avg density error']
                        data_dict_worst_images['worst avg density error'] = img
                    if data['overall F1'] < worst_f1:
                        worst_f1 = data['overall F1']
                        data_dict_worst_images['worst overall F1'] = img
                    if data['defect find F1'] < worst_find_f1:
                        worst_find_f1 = data['defect find F1']
                        data_dict_worst_images['worst defect find F1'] = img

                with open(os.path.join(cfg.OUTPUT_DIR, 'WorstImagesPerStat_TruePredMatch_' + str(true_and_pred_matching_threshold) + '_IoUScoreThresh_' + str(iou_score_threshold_test) + '.json'), 'w') as f:
                    json.dump(data_dict_worst_images, f)

                # Finished one true_and_pred_matching_threshold loop
        # Finished one checkpoint loop
        full_dict_dfs[model_checkpoint] = dict_dfs
        checkpoints_done += 1

    return full_dict_dfs, classification_reports_all_checkpoints_pixels
