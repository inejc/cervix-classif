## Intel & MobileODT Cervical Cancer Screening competition on Kaggle

https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening

### Domain knowledge
* [Anatomy of the cervix](http://www.gfmer.ch/ccdc/pdf/module1.pdf)
* [Cervix types classification](https://kaggle2.blob.core.windows.net/competitions/kaggle/6243/media/Cervix%20types%20clasification.pdf)
* [How to (humanly) recognize cervix types](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/30471)

### Basic dataset stats
* Classify cervix types based on cervical images (three class classification problem)
* Image resolutions between 4128x3096 in 3264x2448 (all jpg - 3 channels)
* 1481 images in the training set (type_1: 250 (0.168805), type_2: 781 (0.527346), type_3: 450 (0.303849))
* 512 images in the test set

#### Additional dataset
* 6924 additional training images (at least prior to fixing)
* [Data cleaning 0](https://www.kaggle.com/chiszpanski/intel-mobileodt-cervical-cancer-screening/non-cervix-images)
* [Data cleaning 1](https://www.kaggle.com/aamaia/intel-mobileodt-cervical-cancer-screening/three-empty-images-in-additional-7z)
* Contains blurry and truncated images

### Useful notebooks
* https://www.kaggle.com/philschmidt/intel-mobileodt-cervical-cancer-screening/cervix-eda/notebook
* https://www.kaggle.com/kambarakun/intel-mobileodt-cervical-cancer-screening/how-to-start-with-python-on-colfax-cluster/notebook
* https://www.kaggle.com/vfdev5/intel-mobileodt-cervical-cancer-screening/data-exploration/notebook
