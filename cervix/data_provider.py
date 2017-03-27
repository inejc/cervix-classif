from os.path import join, dirname

DATA_DIR = join(dirname(dirname(__file__)), 'data')
TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')
SUBMISSIONS_DIR = join(dirname(dirname(__file__)), 'submissions')
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
