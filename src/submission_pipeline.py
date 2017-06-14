from os.path import join

import fire
from keras.applications.inception_v3 import \
    preprocess_input as inception_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess

from average_submissions import create_averaged_submission
from data_provider import SUBMISSIONS_DIR
from resnet50_fine_tune import preprocess_single_input as resnet50_preprocess
from stacking import stack
from utils import create_submission_file
from vgg16_fine_tune import preprocess_single_input as vgg16_preprocess
from vgg19_fine_tune import preprocess_single_input as vgg19_preprocess

GROUPS = [
    {
        'uid': 0,
        'name': 'stable',
        'width': 299,
        'height': 299,
        'models': [
            ('xception_fine_tuned_stable_frozen_86_dropout_0_2_val_loss_0_7288.h5', xception_preprocess),
            ('inception_fine_tuned_stable_frozen_280_dropout_0_5_val_loss_0_7203.h5', inception_preprocess),
            ('resnet50_fine_tuned_stable_frozen_120_dropout_0_5_val_loss_0_7174.h5', resnet50_preprocess),
            ('vgg19_fine_tuned_stable_frozen_17_penultimate_256_dropout_0_5_val_loss_0_6631.h5', vgg19_preprocess),
            ('vgg16_fine_tuned_stable_frozen_6_penultimate_256_dropout_0_5_val_loss_0_6863.h5', vgg16_preprocess),
        ]
    },
    {
        'uid': 1,
        'name': 'stable',
        'width': 299,
        'height': 299,
        'models': [
            ('xception_fine_tuned_stable_frozen_96_dropout_0_6_val_loss_0_7383.h5', xception_preprocess),
            ('inception_fine_tuned_stable_frozen_270_dropout_0_5_val_loss_0_7166.h5', inception_preprocess),
            ('resnet50_fine_tuned_stable_frozen_130_dropout_0_5_val_loss_0_6868.h5', resnet50_preprocess),
            ('vgg19_fine_tuned_stable_frozen_12_penultimate_512_dropout_0_5_val_loss_0_6995.h5', vgg19_preprocess),
            ('vgg16_fine_tuned_stable_frozen_11_penultimate_512_dropout_0_5_val_loss_0_7298.h5', vgg16_preprocess),
        ]
    },
    {
        'uid': 2,
        'name': 'stable',
        'width': 299,
        'height': 299,
        'models': [
            ('xception_fine_tuned_stable_frozen_86_dropout_0_6_val_loss_0_7386.h5', xception_preprocess),
            ('inception_fine_tuned_stable_frozen_260_dropout_0_5_val_loss_0_7440.h5', inception_preprocess),
            ('resnet50_fine_tuned_stable_frozen_140_dropout_0_5_val_loss_0_7365.h5', resnet50_preprocess),
            ('vgg19_fine_tuned_stable_frozen_7_penultimate_512_dropout_0_5_val_loss_0_6881.h5', vgg19_preprocess),
            ('vgg16_fine_tuned_stable_frozen_11_penultimate_256_dropout_0_5_val_loss_0_7105.h5', vgg16_preprocess),
        ]
    },
    {
        'uid': 3,
        'name': 'stable',
        'width': 299,
        'height': 299,
        'models': [
            ('xception_fine_tuned_stable_frozen_86_dropout_0_5_val_loss_0_7520.h5', xception_preprocess),
            ('inception_fine_tuned_stable_frozen_250_dropout_0_5_val_loss_0_7473.h5', inception_preprocess),
            ('resnet50_fine_tuned_stable_frozen_150_dropout_0_5_val_loss_0_7410.h5', resnet50_preprocess),
            ('vgg19_fine_tuned_stable_frozen_7_penultimate_256_dropout_0_5_val_loss_0_7142.h5', vgg19_preprocess),
            ('vgg16_fine_tuned_stable_frozen_6_penultimate_512_dropout_0_5_val_loss_0_7330.h5', vgg16_preprocess),
        ]
    },
]


def make_submission():
    submissions = {}

    for group in GROUPS:
        te_names, te_preds = stack(group)

        submission_name = 'stacked_group_{:d}.csv'.format(group['uid'])
        submissions[submission_name] = 1
        submission_file = join(SUBMISSIONS_DIR, submission_name)
        create_submission_file(te_names, te_preds, submission_file)

    create_averaged_submission(submissions)


if __name__ == '__main__':
    fire.Fire()
