import torch

from src.utils.logger import setup_logger
from src.metrics import MetricLogger, SmoothedValue


logger = setup_logger('engine', save_dir='../logs')


class Trainer(object):
    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, lr_scheduler, metric, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metric = metric
        self.device = device

        self.status = {}

    def train_epoch(self):
        self.model.train()
        metric_logger = MetricLogger(logger=logger, delimiter=" ")
        metric_logger.add_meter('loss', SmoothedValue(window_size=len(self.train_loader), fmt='{value:.6f}'))
        metric_logger.add_meter('acc', SmoothedValue(window_size=len(self.train_loader), fmt='{value:.4f}'))
        header = 'Epoch: [{}]'.format(self.status['epoch'])
        print_freq = 10

        for data_iter_step, (images, labels) in enumerate(metric_logger.log_every(self.train_loader,
                                                                                  print_freq, header)):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            categorical_probs = self.model(images)
            loss = self.criterion(categorical_probs, labels)

            loss.backward()
            self.optimizer.step()

            metric_logger.update(loss=loss.item())
            metric_logger.update(acc=self.metric(categorical_probs, labels))

        self.lr_scheduler.step()
        if self.status['lr'] != self.lr_scheduler.get_last_lr():
            print("Learning rate is changed to {}".format(self.status['lr']))
        self.status['lr'] = self.lr_scheduler.get_last_lr()
        print("Averaged stats:", metric_logger)

    def val_epoch(self):
        metric_logger = MetricLogger(logger=logger, delimiter=" ")
        metric_logger.add_meter('loss', SmoothedValue(window_size=len(self.train_loader), fmt='{value:.6f}'))
        metric_logger.add_meter('acc', SmoothedValue(window_size=len(self.train_loader), fmt='{value:.2f}'))
        header = "Test: "

        self.model.eval()

        with torch.no_grad():
            for images, labels in metric_logger.log_every(self.val_loader, 10, header):
                images = images.to(self.device)
                labels = labels.to(self.device)
                categorical_probs = self.model(images)

                loss = self.criterion(categorical_probs, labels)
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc'].update(self.metric(categorical_probs, labels))

        self.status['val_acc'] = metric_logger.acc.global_average
        print('* Acc {acc.global_average:.3f} loss {loss.global_average:.3f}'.format(
            acc=metric_logger.acc, loss=metric_logger.loss
        ))


def do_train(cfg, model, checkpoint_saver, train_loader, val_loader, criterion,
             optimizer, lr_scheduler, metric):

    engine = Trainer(model, train_loader, val_loader, criterion, optimizer, lr_scheduler,
                     metric, cfg.SOLVER.DEVICE)
    epochs = cfg.SOLVER.NUM_EPOCHS

    engine.status.update({
        'epoch': 0,
        'steps_per_train_epoch': len(train_loader),
        'steps_per_val_epoch': len(val_loader),
        'lr': cfg.OPTIM.LR
    })

    for epoch_id in range(epochs):
        engine.status['epoch'] += 1

        engine.status['mode'] = 'train'
        engine.train_epoch()

        engine.status['mode'] = 'eval'
        engine.val_epoch()

        checkpoint_saver.save_checkpoint(epoch=engine.status['epoch'],
                                         metric=engine.status['val_acc'])


if __name__ == '__main__':
    pass
