import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from .mobilenet_v2 import MobileNetV2, InvertedResidual
from .ssd import SSD, GraphPath
from .predictor import Predictor
from ..utils.config import cfg


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        nn.ReLU6(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_mobilenetv2_ssd_lite(num_classes, width_mult=1.0, is_test=False):
    '''
    MobilenetV2をベースとして、SSDをよりmobileフレンドリーにしたSSDLiteを使用します。

    base_net: ベースネット（MobilenetV2)
    source_layer_indexes:
    extras: SSDのextraモジュールを実装


    '''
    base_net = MobileNetV2(width_mult=width_mult).features

    source_layer_indexes = [
        GraphPath(14, 'conv', 3),
        19,
    ]
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=cfg)


def create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, cfg.image_size, cfg.image_mean,
                          cfg.image_std,
                          iou_threshold=cfg.iou_threshold,
                          candidate_size=candidate_size,
                          device=device)
    return predictor
