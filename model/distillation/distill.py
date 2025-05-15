import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from .distiller import DetectionDistiller


if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'MPD-Net.yaml',
        'data':'./dataset/data.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 32,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        # 'amp': False, #
        'project':'runs/distill',
        'name':'',
        
        # distill
        'prune_model': False,
        'teacher_weights': 'runs/train/exp/weights/best.pt',
        'teacher_cfg': '',
        'kd_loss_type': 'logical',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '12,15,18,21',
        'student_kd_layers': '12,15,18,21',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)

    model.distill()