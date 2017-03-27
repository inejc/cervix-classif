## Intel & MobileODT Cervical Cancer Screening competition on Kaggle

### Domain knowledge
* [Anatomy of the cervix](http://www.gfmer.ch/ccdc/pdf/module1.pdf)
* [Cervix types classification](https://kaggle2.blob.core.windows.net/competitions/kaggle/6243/media/Cervix%20types%20clasification.pdf)
* [How to (humanly) recognize cervix types](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/30471)

### Basic dataset stats
* Classify cervix types based on cervical images (three class classification problem)
* Image resolutions between 4128x3096 in 3264x2448 (all jpg - 3 channels)
* 1481 images in the training set (type_1: 250 (0.168805), type_2: 781 (0.527346), type_3: 450 (0.303849))
* 512 images in the test set
* 6924 additional training images (labeling errors, see https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/30621)

### Useful notebooks
* https://www.kaggle.com/philschmidt/intel-mobileodt-cervical-cancer-screening/cervix-eda/notebook
* https://www.kaggle.com/kambarakun/intel-mobileodt-cervical-cancer-screening/how-to-start-with-python-on-colfax-cluster/notebook
* https://www.kaggle.com/vfdev5/intel-mobileodt-cervical-cancer-screening/data-exploration/notebook

### Project setup
```
git clone https://github.com/inejc/cervix-classif.git
cd cervix-classif
make setup
. env/bin/activate
```

### Training and validation split
Be careful with this on server since your data directory is just a symlink to everyone's shared directory :). Run `python data_dirs_organizer.py organize` to split the dataset into training and validation sets. Currently each image's smallest dimension is resized to 299 pixels and then cropped at the center of the larger dimension (obtaining 299x299 images). We can call the detection functionality from here later. Similarly run `python data_dirs_organizer.py clean` to delete all resized and organized images (i.e. to undo organize).
