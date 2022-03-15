from yacs.config import CfgNode as CN


_C = CN()

# Data paths
_C.DATA = CN()

_C.DATA.TRAIN_ANNOTATIONS = ['../data/FOLD_1_train.json', '../data/FOLD_2_train.json',
                             '../data/FOLD_3_train.json', '../data/FOLD_4_train.json',
                             '../data/FOLD_5_train.json']
_C.DATA.TEST_ANNOTATIONS = ['../data/FOLD_1_test.json', '../data/FOLD_2_test.json',
                            '../data/FOLD_3_test.json', '../data/FOLD_4_test.json',
                            '../data/FOLD_5_test.json']

# Augmentation
_C.AUGMENTATION = CN()

_C.AUGMENTATION.CUTOUT = CN()
_C.AUGMENTATION.CUTOUT.MASK_SIZE = 15
_C.AUGMENTATION.CUTOUT.PROB = 0.5
_C.AUGMENTATION.CUTOUT.CUTOUT_INSIDE = True
_C.AUGMENTATION.CUTOUT.MASK_COLOR = 0

_C.AUGMENTATION.RESIZE = CN()
_C.AUGMENTATION.RESIZE.SIZE = (120, 160)

_C.AUGMENTATION.NORMALIZE = CN()
_C.AUGMENTATION.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
_C.AUGMENTATION.NORMALIZE.STD = [0.229, 0.224, 0.225]

# Data loader
_C.DATA_LOADER = CN()
_C.DATA_LOADER.TRAIN_BATCH_SIZE = 4
_C.DATA_LOADER.TEST_BATCH_SIZE = 4
_C.DATA_LOADER.NUM_WORKERS = 2

# Model
_C.MODEL = CN()

_C.MODEL.NAME = 'efficientnet_b0'
_C.MODEL.PRETRAINED = False

# Loss
_C.LOSS = CN()

_C.LOSS.NAME = 'am_softmax'
_C.LOSS.IN_FEATURES = 1280
_C.LOSS.OUT_FEATURES = 9
_C.LOSS.S = 30.0
_C.LOSS.M = 0.4
_C.LOSS.LOSS_RATIO = 1

# Optim
_C.OPTIM = CN()

_C.OPTIM.OPTIMIZER = 'adam'
_C.OPTIM.LR_SCHEDULER = 'step_lr'
_C.OPTIM.LR = 1e-4
_C.OPTIM.LR_DECAY_STEP = 30

# Solver
_C.SOLVER = CN()
_C.SOLVER.NUM_EPOCHS = 60
_C.SOLVER.DEVICE = 'cpu'
_C.SOLVER.LOG_ITER = 1
_C.SOLVER.CHECKPOINT_DIR = '../checkpoints'


def get_cfg_defaults():

    return _C.clone()
