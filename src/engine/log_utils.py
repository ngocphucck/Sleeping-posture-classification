import datetime


def log(logger, cfg, status):
    mode = status['mode']
    if mode == 'train':
        epoch_id = status['epoch_id']
        step_id = status['step_id']
        steps_per_epoch = status['steps_per_epoch']
        # batch_time = status['batch_time']
        # data_time = status['data_time']
        # epochs = cfg['epochs']
        # batch_size = cfg['batch_size']

        space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
        if step_id % cfg.SOLVER.LOG_ITER == 0:
            # eta_steps = (epochs - epoch_id) * steps_per_epoch - step_id
            # eta_sec = eta_steps * batch_time.global_avg
            # eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            # ips = float(batch_size) / batch_time.avg
            fmt = ' '.join([
                'Epoch: [{}]',
                '[{' + space_fmt + '}/{}]',
                'learning_rate: {lr:.6f}',
                # '{meters}',
                # 'eta: {eta}',
                # 'batch_cost: {btime}',
                # 'data_cost: {dtime}',
                # 'ips: {ips:.4f} images/s',
                ])
            fmt = fmt.format(
                epoch_id,
                step_id,
                steps_per_epoch,
                lr=status['learning_rate'])
            logger.info(fmt)
    if mode == 'eval':
        step_id = status['step_id']
        if step_id % 100 == 0:
            logger.info("Eval iter: {}".format(step_id))
