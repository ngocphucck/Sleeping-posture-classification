from .amsoftmax import AMSoftmax


def make_loss_fn(cfg):

    return AMSoftmax(cfg.LOSS.IN_FEATURES, cfg.LOSS.OUT_FEATURES, cfg.LOSS.S, cfg.LOSS.M)
