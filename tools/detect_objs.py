import _init_paths
import os
import json
import cv2
import h5py
import caffe
import numpy as np
import cPickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.nms_wrapper import nms

# detected objects save path
coco_obj_save_path = '../data/coco_objs'
train_id_path = os.path.join(coco_obj_save_path, 'train_box_imgid2idx.pkl')
val_id_path = os.path.join(coco_obj_save_path, 'val_box_imgid2idx.pkl')
train_feat_path = os.path.join(coco_obj_save_path, 'train_box.h5')
val_feat_path = os.path.join(coco_obj_save_path, 'val_box.h5')
if not os.path.exists(coco_obj_save_path):
    os.makedirs(coco_obj_save_path)

# Dataset Setting
coco_img_path = '/home/cl/Dataset/COCO/'
coco_anno_path = '/home/cl/Dataset/COCO/annotations'

# Caffe Setting
gpu_id = 1
caffe_proto = '../models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
caffe_model = '../data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel'
cfg_path = '../experiments/cfgs/faster_rcnn_end2end_resnet.yml'
cfg_from_file(cfg_path)
# cfg.TEST.HAS_RPN True means no extra bounding box, False means with extra bounding box input

# Faster-RCNN Setting
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TEST_SCALES = [600]
MAX_SIZE = 1000
DEDUP_BOXES = 0.0625
CONF_THRESH = 0.2
MIN_BOXES = 36
MAX_BOXES = 36

coco_train_anno_path = os.path.join(coco_anno_path, 'captions_train2014.json')
coco_val_anno_path = os.path.join(coco_anno_path, 'captions_val2014.json')
coco_train_img_path = os.path.join(coco_img_path, 'train2014')
coco_val_img_path = os.path.join(coco_img_path, 'val2014')

def assert_exist_path(path):
    assert os.path.exists(path)

assert_exist_path(caffe_proto)
assert_exist_path(caffe_model)
    
assert_exist_path(coco_train_anno_path)
assert_exist_path(coco_val_anno_path)
assert_exist_path(coco_train_img_path)
assert_exist_path(coco_val_img_path)

def filter_image_info(split_name):
    split = []
    if split_name == 'train':
        with open(coco_train_anno_path) as f:
            data  = json.load(f)
            for item in tqdm(data['images']):
                image_id = int(item['id'])
                image_width = int(item['width'])
                image_height = int(item['height'])
                image_path = os.path.join(coco_train_img_path, item['file_name'])
                assert_exist_path(image_path)
                split.append({'id': image_id, 'width': image_width, 
                             'height': image_height, 'filename': image_path})
    elif split_name == 'val':
        with open(coco_val_anno_path) as f:
            data  = json.load(f)
            for item in tqdm(data['images']):
                image_id = int(item['id'])
                image_width = int(item['width'])
                image_height = int(item['height'])
                image_path = os.path.join(coco_val_img_path, item['file_name'])
                assert_exist_path(image_path)
                split.append({'id': image_id, 'width': image_width, 
                             'height': image_height, 'filename': image_path})
    else:
        print 'Unknow split'
    return split

train_img_info = filter_image_info('train')
val_img_info = filter_image_info('val')

## https://github.com/zjuchenlong/bottom-up-attention/blob/master/lib/utils/blob.py
def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

## https://github.com/zjuchenlong/bottom-up-attention/blob/master/lib/fast_rcnn/test.py
def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > MAX_SIZE:
            im_scale = float(MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois=None):
    """
    Convert an image and RoIs within that image into network inputs.
    boxes (ndarray): R x 4 array of object proposals
    """
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

## https://github.com/zjuchenlong/bottom-up-attention/blob/master/lib/fast_rcnn/bbox_transform.py
def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """ Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def boxes_to_spatial(bboxes, image_w, image_h):
    box_width = bboxes[:, 2] - bboxes[:, 0]
    box_height = bboxes[:, 3] - bboxes[:, 1]
    scaled_width = box_width / image_w
    scaled_height = box_height / image_h
    scaled_x = bboxes[:, 0] / image_w
    scaled_y = bboxes[:, 1] / image_h

    box_width = box_width[..., np.newaxis]
    box_height = box_height[..., np.newaxis]
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]

    spatial_features = np.concatenate(
        (scaled_x,
         scaled_y,
         scaled_x + scaled_width,
         scaled_y + scaled_height,
         scaled_width,
         scaled_height),
        axis=1)
    return spatial_features

caffe.set_mode_gpu()
caffe.set_device(gpu_id)
caffe_net = caffe.Net(caffe_proto, caffe.TEST, weights=caffe_model)

for split_name in ['train', 'val']:
    split_img_info = eval(split_name + '_img_info')
    split_imgid2idx = {}
    split_feat_file = h5py.File(eval(split_name + '_feat_path'), 'w')
    assert MIN_BOXES == MAX_BOXES
    num_fixed_boxes = MIN_BOXES
    split_img_bb = split_feat_file.create_dataset('image_bb', (len(split_img_info), num_fixed_boxes, 4), 'f')

    for idx, img in tqdm(enumerate(split_img_info)):
        im = cv2.imread(img['filename'])
        blobs, im_scales = _get_blobs(im)
        
        # When mapping from image ROIs to feature map ROIs, there's some aliasing (some distinct image ROIs get mapped 
        # to the same feature ROI). Here, we identify duplicate feature ROIs, so we only compute features on the unique subset.
        if DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'] * DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)
            blobs['rois'] = blobs['rois'][index, :]
            boxes = boxes[index, :]
        
        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)

        # reshape network inputs
        caffe_net.blobs['data'].reshape(*(blobs['data'].shape))
        if 'im_info' in caffe_net.blobs:
            caffe_net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        if not cfg.TEST.HAS_RPN:
            caffe_net.blobs['rois'].reshape(*(blobs['rois'].shape))

        # do forward
        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
        if 'im_info' in caffe_net.blobs:
            forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
        if not cfg.TEST.HAS_RPN:
            forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
        blobs_out = caffe_net.forward(**forward_kwargs)

        if cfg.TEST.HAS_RPN:
            assert len(im_scales) == 1, "Only single-image batch implemented"
            rois = caffe_net.blobs['rois'].data.copy()
            # unscale back to raw image space
            boxes = rois[:, 1:5] / im_scales[0]

        scores = blobs_out['cls_prob']

        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

        if DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
            # Map scores and predictions back to the original set of boxes
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]

        cls_boxes = rois[:, 1:5] / im_scales[0]
        pool5 = caffe_net.blobs['pool5_flat'].data

        max_conf = np.zeros((rois.shape[0]))
        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

        keep_boxes = np.where(max_conf >= CONF_THRESH)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

        final_boxes = cls_boxes[keep_boxes]

        final_spatial = boxes_to_spatial(final_boxes, img['width'], img['height'])
        final_feature = pool5[keep_boxes]

        split_imgid2idx[img['id']] = idx
        split_img_bb[idx] = final_boxes

    split_feat_file.close()    
    with open(eval(split_name + '_id_path'), 'wb') as f:
        pkl.dump(split_imgid2idx, f)
    print("done!")
