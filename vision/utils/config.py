import numpy as np
import os
from .box_utils import SSDSpec, SSDBoxSizes, PriorBox, generate_ssd_priors


class Config:
    '''
    ディレクトリパス、ハイパーパラメータ、SSDの設定を格納しています。
    '''
    root_dir = '/home/shal/workspace/mobilenetv2-with-flask'
    VOC_Dir = os.path.join(root_dir, 'data/VOCDevkit/VOC2012/')
    checkpoint_dir = os.path.join(root_dir, 'models/')
    save_loss_dir = os.path.join(root_dir, 'loss/')
    base_net = os.path.join(root_dir, 'models/mb2-imagenet-71_8.pth')
    model_path = os.path.join(root_dir, 'models/mb2-ssd-lite-Epoch-149-Loss-2.65.pth')
    label_path = os.path.join(root_dir, 'models/voc-model-labels.txt')

    image_size = 300
    image_mean = np.array([127, 127, 127])
    image_std = 128.0
    iou_threshold = 0.45
    center_variance = 0.1
    size_variance = 0.2
    # feature_map, step, SSDBoxSizes(min, max), aspect_ratios
    specs = [
        SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
        SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
        SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
        SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
        SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
        SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
    ]

    # priors = PriorBox()
    priors = generate_ssd_priors(specs, image_size)
    mb2_width_mult = 1.0
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    scheduler = 'cosine'  # {cosine, multi-step}
    num_epochs = 300
    validation_epochs = 1
    t_max = 200
    batch_size = 32

    img_array_limmit = 300



cfg = Config()

