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

#### Additional dataset
* 6924 additional training images (at least prior to fixing)
* [Data cleaning 0](https://www.kaggle.com/chiszpanski/intel-mobileodt-cervical-cancer-screening/non-cervix-images)
* [Data cleaning 1](https://www.kaggle.com/aamaia/intel-mobileodt-cervical-cancer-screening/three-empty-images-in-additional-7z)
* Contains blurry and truncated images

### Useful notebooks
* https://www.kaggle.com/philschmidt/intel-mobileodt-cervical-cancer-screening/cervix-eda/notebook
* https://www.kaggle.com/kambarakun/intel-mobileodt-cervical-cancer-screening/how-to-start-with-python-on-colfax-cluster/notebook
* https://www.kaggle.com/vfdev5/intel-mobileodt-cervical-cancer-screening/data-exploration/notebook

### Project setup

#### Initial setup
This should be performed only once (unless you change `Dockerfile` then you
have to run `make setup-docker` again.

```
git clone https://github.com/inejc/cervix-classif.git
make setup-host
make setup-docker
```

Since mounting of data folder assumes specific path, it's best if you clone `cervix-classif`
repository directly into your home directory on `ocean`. If you have a sudden
change of heart, you can change mounted directory in `docker-compose.yml` file.

#### Everytime you start working
At the start of the day, you should run `make run`. This runs docker container
in the background. You can see if it is running by running `docker ps`. By
default it's named after your `$USER` on the server.

To login to the container, you can run `make cli` and bash will open. Your
home directory from `ocean` server is directly mounted into docker home
directory (`/home/user`). Data is also automatically mounted on
`/home/user/cervix-classif/data`.

There's a shortcut with running any command with `make r cmd='<your command>'.
However we noticed some issues with that, you're better off just using the above
approach.

#### Additional commands
`make stop` - stops any docker container you might be running.

### Using GPUs
Sharing is caring! Limit your process to your own GPU. This can be done with the
`CUDA_VISIBLE_DEVICES=id` flag when using TensorFlow and with `THEANO_FLAGS=device=gpuid`
when using Theano. To see which GPUs are currently free or in use, you can use
`nvidia-smi` to get an overview of running processes.

### Training and validation split
Be careful with this on server since your data directory is just a symlink to everyone's shared directory :). Run `python data_dirs_organizer.py organize` to split the dataset into training and validation sets. Currently each image's smallest dimension is resized to 299 pixels and then cropped at the center of the larger dimension (obtaining 299x299 images). We can call the detection functionality from here later. Similarly run `python data_dirs_organizer.py clean` to delete all resized and organized images (i.e. to undo organize).
