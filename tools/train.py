from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.engine import do_train
from src.dataset import make_loader
from src.losses import make_loss_fn
from src.models import make_model
from src.metrics import AccuracyMetric
from configs import get_cfg_defaults


def main(cfg):
    model = make_model(cfg)
    criterion1 = make_loss_fn(cfg)
    criterion2 = CrossEntropyLoss()

    train_loader = make_loader(cfg, fold_number=1, mode='train')
    val_loader = make_loader(cfg, fold_number=1, mode='val')
    optimizer = Adam(model.parameters(), lr=cfg.SOLVER.LR)
    metric = AccuracyMetric()

    do_train(cfg, model, train_loader, val_loader, criterion1, criterion2, optimizer, metric)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    main(cfg)
    pass

