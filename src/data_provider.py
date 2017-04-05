from json import load, dump
from os import listdir
from os.path import join, dirname, isfile

DATA_DIR = join(dirname(dirname(__file__)), 'data')
TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')
ADDITIONAL_DIR = join(DATA_DIR, 'additional')
SUBMISSIONS_DIR = join(dirname(dirname(__file__)), 'submissions')
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
EXPERIMENTS_DIR = join(dirname(dirname(__file__)), 'experiments')

CLASSES = ['Type_1', 'Type_2', 'Type_3']
ORGANIZED_DATA_INFO_FILE = 'organized_data_info_.json'

IMAGES_BLACKLIST_FILE = join(
    dirname(dirname(__file__)),
    'src', 'images_blacklist.txt'
)


def load_organized_data_info(imgs_dim, name=''):
    """Loads the train, val, test datasets info file.
    
    Returns
    -------
    dict
        'dir_tr': absolute path of the training directory
        'num_tr': number of training images
        'dir_val': absolute path of the validation directory
        'num_val': number of validation images
        'dir_te': absolute path of the test directory
        'num_te': number of test images
        'num_classes': number of distinct classes
    """
    if not isfile(organized_data_info_file(imgs_dim, name)):
        raise FileNotFoundError('run data_dirs_organizer.py organize first')
    with open(organized_data_info_file(imgs_dim, name), 'r') as f:
        return load(f)


def save_organized_data_info(info, imgs_dim, name=''):
    with open(organized_data_info_file(imgs_dim, name), 'w') as f:
        dump(info, f)


def organized_data_info_file(imgs_dim, name):
    split = ORGANIZED_DATA_INFO_FILE.split('.')
    split[0] += str(imgs_dim) + '_{:s}'.format(name)
    return join(DATA_DIR, '.'.join(split))


def num_examples_per_class_in_dir(dir_):
    """Returns number of images for each CLASSES element (subdir) in the
    dir_ directory."""
    num_per_class = {}

    for class_ in CLASSES:
        class_dir = join(dir_, class_)
        num = len([x for x in listdir(class_dir) if x.endswith('.jpg')])
        num_per_class[class_] = num

    return num_per_class
