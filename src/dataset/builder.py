from torch.utils.data import DataLoader
from torchvision import transforms as T

from .datasets import IRPoseDataset
from .transforms import *
from configs.defaults import get_cfg_defaults


def make_transforms(cfg, mode='train'):
    if mode == 'train':
        transform = T.Compose([
            T.ToPILImage(),
            RandAugment(n=cfg.AUGMENTATION.RAND_N, m=cfg.AUGMENTATION.RAND_M),
            ColorConvert(),
            Resize(cfg.AUGMENTATION.RESIZE.SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.AUGMENTATION.NORMALIZE.MEAN,
                        std=cfg.AUGMENTATION.NORMALIZE.STD)
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            ColorConvert(),
            Resize(cfg.AUGMENTATION.RESIZE.SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.AUGMENTATION.NORMALIZE.MEAN,
                        std=cfg.AUGMENTATION.NORMALIZE.STD)
        ])

    return transform


def make_loader(cfg, fold_number=1, mode='train'):
    transforms = make_transforms(cfg, mode=mode)
    num_workers = cfg.DATA_LOADER.NUM_WORKERS

    if mode == 'train':
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
        shuffle = True
        dataset = IRPoseDataset(transforms, cfg.DATA.TRAIN_ANNOTATIONS[fold_number - 1])
    else:
        batch_size = cfg.DATA_LOADER.TEST_BATCH_SIZE
        shuffle = False
        dataset = IRPoseDataset(transforms, cfg.DATA.TEST_ANNOTATIONS[fold_number - 1])

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    loader = make_loader(cfg)
    pass
