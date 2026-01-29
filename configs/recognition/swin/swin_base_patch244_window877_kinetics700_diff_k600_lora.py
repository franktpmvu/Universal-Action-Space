_base_ = "swin_base_patch244_window877_kinetics400_22k.py"

data_root = 'data/kinetics700_diff/train'
data_root_val = 'data/kinetics700_diff/val'
ann_file_train = 'data/kinetics700_diff/k700_diff_train.txt'
ann_file_val = 'data/kinetics700_diff/k700_diff_val.txt'
ann_file_test = 'data/kinetics700_diff/k700_diff_val.txt'


data = dict(
    train=dict(
        ann_file=ann_file_train,
        data_prefix=data_root),
    val=dict(
        ann_file=ann_file_val,
        data_prefix=data_root_val),
    test=dict(
        ann_file=ann_file_test,
        data_prefix=data_root_val))

# model=dict(cls_head=dict(num_classes=103))
model=dict(cls_head=dict(num_classes=103), backbone=dict(use_lora=True, lora_rank=32, lora_alpha=None))

freeze_backbone = True

load_from = 'checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
