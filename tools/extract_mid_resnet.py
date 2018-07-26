import _init_paths
import os
import cv2
import json
import h5py
import caffe
import numpy as np
from tqdm import tqdm
import cPickle as pkl
from fast_rcnn.config import cfg, cfg_from_file

INPUT_SIZE = 224
gpu_id = 1
# Faster-RCNN Setting
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# MAX_SIZE = 1000
# DEDUP_BOXES = 0.0625
DEDUP_BOXES = 0 # Keep all the 36 bounding box
CONF_THRESH = 0.2
MIN_BOXES = 36
MAX_BOXES = 36

# load path
coco_obj_path = '../data/coco_objs'
coco_img_path = '../data/coco'
coco_anno_path = '../data/coco/annotations'

# Caffe Setting
gpu_id = 0
caffe_proto = '../models/vg/ResNet-101/faster_rcnn_end2end_final/mid_test_gt.prototxt'
# caffe_model = '../data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel'
caffe_model = '../data/imagenet_models/ResNet-101-model.caffemodel'
cfg_path = '../experiments/cfgs/faster_rcnn_end2end_resnet.yml'
cfg_from_file(cfg_path)

cfg.TEST.HAS_RPN = False

train_obj_path = os.path.join(coco_obj_path, 'train_obj.pkl')
val_obj_path = os.path.join(coco_obj_path, 'val_obj.pkl')
coco_train_anno_path = os.path.join(coco_anno_path, 'captions_train2014.json')
coco_val_anno_path = os.path.join(coco_anno_path, 'captions_val2014.json')
coco_train_img_path = os.path.join(coco_img_path, 'train2014')
coco_val_img_path = os.path.join(coco_img_path, 'val2014')

# save_path
coco_mid_save_path = '../data/coco_mid_grid_objs_224'
if not os.path.exists(coco_mid_save_path):
    os.makedirs(coco_mid_save_path)
coco_train_save_id = os.path.join(coco_mid_save_path, 'train_mid_imgid2idx.pkl')
coco_val_save_id = os.path.join(coco_mid_save_path, 'val_mid_imgid2idx.pkl')
coco_train_objs_feat = os.path.join(coco_mid_save_path, 'train_mid_objs.h5')
coco_val_objs_feat = os.path.join(coco_mid_save_path, 'val_mid_objs.h5')
coco_train_grid_feat = os.path.join(coco_mid_save_path, 'train_mid_grid.h5')
coco_val_grid_feat = os.path.join(coco_mid_save_path, 'val_mid_grid.h5')


def assert_exist_path(path):
    assert os.path.exists(path), 'This is no %s'%(path)

assert_exist_path(train_obj_path)
assert_exist_path(val_obj_path)
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
        print('Unknow split: %s'%(split_name))
    return split

train_img_info = filter_image_info('train')
val_img_info = filter_image_info('val')

with open(train_obj_path, 'r') as f:
    train_obj = pkl.load(f)

with open(val_obj_path, 'r') as f:
    val_obj = pkl.load(f)

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

    height, width, rgb = im_orig.shape 
    im_size_min = min(height, width)
    im_size_max = max(height, width)

    processed_ims = []

    im_scale = float(INPUT_SIZE) / float(im_size_min)
    # # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > MAX_SIZE:
    #     im_scale = float(MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    crop_offset = int((im_size_max - im_size_min)/2)

    if height < width:
        im = im[:, crop_offset : crop_offset + INPUT_SIZE]
    elif height > width:
        im = im[crop_offset: crop_offset + INPUT_SIZE, :]
    else:
        pass
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scale, crop_offset

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
    blobs['data'], im_scale, crop_offset = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, np.array([im_scale]))
    return blobs, im_scale, crop_offset

caffe.set_mode_gpu()
caffe.set_device(gpu_id)
caffe_net = caffe.Net(caffe_proto, caffe.TEST, weights=caffe_model)

for split_name in ['train', 'val']:
# for split_name in ['train']:
# for split_name in ['val']:
    split_img_info = eval(split_name + '_img_info')

    split_obj = eval(split_name + '_obj')
    split_save_grid_h5_file = h5py.File(eval('coco_%s_grid_feat'%(split_name)), 'w')
    split_save_objs_h5_file = h5py.File(eval('coco_%s_objs_feat'%(split_name)), 'w')
    split_save_grid_feat = split_save_grid_h5_file.create_dataset('image_features',
                                                                  (len(split_img_info), 14*14, 1024), 'f')
    split_save_objs_feat = split_save_objs_h5_file.create_dataset('image_features', (len(split_img_info), 36, 1024), 'f')
    split_save_id = {}
    print('Total include %d images'%len(split_img_info))
    for idx, img in tqdm(enumerate(split_img_info)):
        img_id = img['id']
        split_save_id[img_id] = idx
        im = cv2.imread(img['filename'])
        box_orig = split_obj[img_id]

        # clean check
        assert np.all(box_orig[:, 0] <= box_orig[:, 2])
        assert np.all(box_orig[:, 1] <= box_orig[:, 3])
        assert box_orig[:, 2].max() <= im.shape[1]
        assert box_orig[:, 3].max() <= im.shape[0]
    
        blobs, im_scale, crop_offset = _get_blobs(im, rois=box_orig)

        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scale]], dtype=np.float32)

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

        # pool5 = caffe_net.blobs['pool5_flat'].data
        res4b22 = caffe_net.blobs['res4b22'].data

        # scale and crop box
        box_scale = box_orig * im_scale
        box_crop = box_scale.copy()
        if img['width'] > img['height']:
            box_crop[:, 0] = np.clip(box_scale[:, 0], a_min=crop_offset, a_max=crop_offset + INPUT_SIZE)
            box_crop[:, 2] = np.clip(box_scale[:, 2], a_min=crop_offset, a_max=crop_offset + INPUT_SIZE)
            box_crop[:, 0] -= crop_offset
            box_crop[:, 2] -= crop_offset
        elif img['width'] < img['height']:
            box_crop[:, 1] = np.clip(box_scale[:, 1], a_min=crop_offset, a_max=crop_offset + INPUT_SIZE)
            box_crop[:, 3] = np.clip(box_scale[:, 3], a_min=crop_offset, a_max=crop_offset + INPUT_SIZE)
            box_crop[:, 1] -= crop_offset
            box_crop[:, 3] -= crop_offset
        else:
            pass

        # clean check
        assert np.all(box_crop[:, 0] <= box_crop[:, 2])
        assert np.all(box_crop[:, 1] <= box_crop[:, 3])
        assert box_crop.max() <= INPUT_SIZE
        assert box_crop.min() >= 0         

        # 600 * 600 -> 38 * 38
        # 224 * 224 -> 14 * 14
        assert INPUT_SIZE == 224
        box_crop = np.around(box_crop / 16.0).astype(np.int32)
        # position -> mask
        mask = np.zeros((36, 14, 14))
        for i, each_box in enumerate(box_crop):
            mask[i][each_box[1]: each_box[3], each_box[0]: each_box[2]] = 1
        mask = mask.reshape(36, 14*14)
        mask = mask / (mask.sum(1, keepdims=1) + 1e-8)

        each_img_feat = res4b22[0].reshape(1024, 14*14).transpose(1, 0)
        box_img_feat = np.matmul(mask, each_img_feat)

        split_save_grid_feat[idx] = each_img_feat
        split_save_objs_feat[idx] = box_img_feat

    with open(eval('coco_%s_save_id'%(split_name)), 'wb') as f:
        pkl.dump(split_save_id, f)

    split_save_grid_h5_file.close()
    split_save_objs_h5_file.close()
    print('Done')
