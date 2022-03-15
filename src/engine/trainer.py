import torch
from tqdm import tqdm

from src.utils.logger import setup_logger
from .helpers import log


logger = setup_logger('engine')


class Trainer(object):
    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, metric, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.device = device

        self.status = {}

    def train_epoch(self):
        self.model.train()
        train_epoch_iterator = tqdm(self.train_loader,
                                    desc="Training (X / X Steps) (loss=X.X)",
                                    bar_format="{l_bar}{r_bar}",
                                    dynamic_ncols=True)
        for images, labels in train_epoch_iterator:
            self.status['step_train_id'] += 1
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            categorical_probs = self.model(images)
            loss = self.criterion(categorical_probs, labels)

            loss.backward()
            self.status['loss'] = loss.item()
            self.optimizer.step()
            train_epoch_iterator.set_description(
                "Training (Epoch %d) (loss=%2.5f)" % (self.status['epoch_id'], self.status['loss'])
            )

            self.metric.update(categorical_probs, labels)

    def val_epoch(self):
        self.model.eval()
        val_epoch_iterator = tqdm(self.val_loader,
                                  desc="Validating (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)

        with torch.no_grad():
            for images, labels in val_epoch_iterator:
                self.status['step_val_id'] += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                categorical_probs = self.model(images)

                loss = self.criterion(categorical_probs, labels)
                self.status['loss'] = loss.item()
                val_epoch_iterator.set_description(
                    "Validating (Epoch %d) (loss=%2.5f)" % (self.status['epoch_id'], self.status['loss'])
                )
                self.metric.update(categorical_probs, labels)


def do_train(cfg, model, checkpoint_saver, train_loader, val_loader, criterion,
             optimizer, metric):

    engine = Trainer(model, train_loader, val_loader, criterion, optimizer,
                     metric, cfg.SOLVER.DEVICE)
    epochs = cfg.SOLVER.NUM_EPOCHS

    engine.status.update({
        'epoch_id': 0,
        'step_id': 0,
        'step_train_id': 0,
        'step_val_id': 0,
        'steps_per_train_epoch': len(train_loader),
        'steps_per_val_epoch': len(val_loader),
        'learning_rate': cfg.SOLVER.LR
    })

    for epoch_id in range(epochs):
        engine.status['epoch_id'] += 1
        engine.status['step_id'] += 1

        engine.status['mode'] = 'train'
        log(logger, cfg, engine.status)
        engine.train_epoch()
        metric.accumulate()
        engine.status['train_accuracy'] = metric.get_results()
        logger.info('Training: ' + metric.log())
        metric.reset()

        engine.status['mode'] = 'eval'
        log(logger, cfg, engine.status)
        engine.val_epoch()
        metric.accumulate()
        engine.status['val_accuracy'] = metric.get_results()
        logger.info('Validating: ' + metric.log(logger))
        metric.reset()

        checkpoint_saver.save_checkpoint(epoch=engine.status['epoch_id'],
                                         metric=engine.status['val_accuracy'])


if __name__ == '__main__':
    pass
