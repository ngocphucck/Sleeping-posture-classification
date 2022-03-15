from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.engine import do_train
from src.dataset import make_loader
from src.utils import create_model
from src.metrics import Accuracy
from configs import get_cfg_defaults
from src.utils import CheckpointSaver
from src.optim import create_optimizer, create_lr_scheduler
from src.models import efficientnet, convnext


def main(cfg):
    model = create_model(cfg.MODEL.NAME, cfg.MODEL.PRETRAINED)
    criterion = CrossEntropyLoss()

    train_loader = make_loader(cfg, fold_number=1, mode='train')
    val_loader = make_loader(cfg, fold_number=1, mode='val')
    optimizer = create_optimizer(name=cfg.OPTIM.OPTIMIZER, parameters=model.parameters(),
                                 lr=cfg.OPTIM.LR)
    lr_scheduler = create_lr_scheduler(name='step_lr', optimizer=optimizer, step_size=cfg.OPTIM.LR_DECAY_STEP)
    checkpoint_saver = CheckpointSaver(model, optimizer, checkpoint_dir='../checkpoints')
    metric = Accuracy()

    do_train(cfg, model, checkpoint_saver, train_loader, val_loader, criterion, optimizer, lr_scheduler, metric)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    main(cfg)
    pass

