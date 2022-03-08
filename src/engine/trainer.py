import torch
import os
from tqdm import tqdm

from src.utils.logger import setup_logger
from .log_utils import log


logger = setup_logger('engine')


class Trainer(object):
    def __init__(self, model, train_loader, val_loader,
                 criterion1, criterion2, optimizer, metric, device, loss_ratio=1.0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.optimizer = optimizer
        self.loss_ratio = loss_ratio
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

            features, categorical_probs = self.model(images)
            loss1 = self.criterion1(features, labels)
            loss2 = self.criterion2(categorical_probs, labels)
            loss = loss1 * self.loss_ratio + loss2

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
                features, categorical_probs = self.model(images)

                loss1 = self.criterion1(features, labels)
                loss2 = self.criterion2(categorical_probs, labels)
                loss = loss1 * self.loss_ratio + loss2

                self.status['loss'] = loss.item()
                val_epoch_iterator.set_description(
                    "Validating (Epoch %d) (loss=%2.5f)" % (self.status['epoch_id'], self.status['loss'])
                )
                self.metric.update(categorical_probs, labels)


def do_train(cfg, model, train_loader, val_loader, criterion1, criterion2,
             optimizer, metric):

    engine = Trainer(model, train_loader, val_loader, criterion1, criterion2, optimizer,
                     metric, cfg.SOLVER.DEVICE, cfg.LOSS.LOSS_RATIO)
    epochs = cfg.SOLVER.NUM_EPOCHS
    best_accuracy = 0

    engine.status.update({
        'epoch_id': 0,
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
        logger.info('Validating: ' + metric.log())
        metric.reset()

        if engine.status['val_accuracy'] > best_accuracy:
            best_accuracy = engine.status['val_accuracy']
            torch.save(model.state_dict(), os.path.join(cfg.SOLVER.CHECKPOINT_DIR, 'best_model.pth'))
            logger.info("Best test accuracy is {:0.3f}.".format(best_accuracy))


if __name__ == '__main__':
    pass
