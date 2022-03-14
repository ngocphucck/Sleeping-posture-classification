import json
import os
from tqdm import tqdm
import yaml
import logging


_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        pass
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict)


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def write_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f)


def make_annotation_data(data_dir, save_dir):
    for fold in tqdm(os.listdir(data_dir)):
        fold_data_dir = os.path.join(data_dir, fold)
        for mode in os.listdir(fold_data_dir):
            data = {}
            image_dir = os.path.join(fold_data_dir, mode)
            for label in os.listdir(image_dir):
                class_dir = os.path.join(image_dir, label)
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    image_path = '/'.join(image_path.split('/')[1:])
                    data[image_path] = int(label) - 1

            save_json_path = os.path.join(save_dir, fold + '_' + mode + '.json')
            with open(save_json_path, 'w') as f:
                json.dump(data, f)


def get_config(yaml_file='../../configs/configs.yml'):
    with open(yaml_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg


if __name__ == '__main__':
    make_annotation_data('../../data/IR_9class_5_FOLD', '../../data')
    # cfg = get_config()
    pass
