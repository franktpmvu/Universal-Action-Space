_base_ = "swin_base_patch244_window877_kinetics400_22k.py"

data_root = 'data/mammalnet/mammalnet'
data_root_val = 'data/mammalnet/mammalnet'
ann_file_train = 'data/mammalnet/train_base.txt'
ann_file_val = 'data/mammalnet/val_base.txt'
ann_file_test = 'data/mammalnet/test_base.txt'

data = dict(
    train=dict(
        ann_file=ann_file_train,
        data_prefix=data_root),
    val=dict(
        ann_file=ann_file_val,
        data_prefix=data_root),
    test=dict(
        ann_file=ann_file_test,
        data_prefix=data_root))

model=dict(cls_head=dict(num_classes=12))

freeze_backbone = True

load_from = 'checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
