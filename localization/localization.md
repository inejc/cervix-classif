# Localization

## ImageJ
We use [ImageJ](https://imagej.nih.gov/ij/index.html) to draw bounding boxes since it's very fast and simple to use. ImageJ by itself isn't that great, but there's a really neat [macro](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/30471#169943) we found on in the Kaggle discussions that makes drawing bouding boxes very simple. The macro is also available in `roi.ijm`. Copy this file to your ImageJ `macros` directory. Next, you have to change the macro save destination for the generated roi files. Open up the .ijm file and set the `roi_path` to some folder on your PC. All the roi files will be stored to that folder.

You're almost done! Now, run ImageJ, select Plugins > Macros > Install... from the menu and find your new macro file there. Note that this must be done everytime you run ImageJ. Now you can start tagging. Open images that haven't yet been tagged (the images that have already been tagged should be on ocean `/mnt/nfs/kaggle/cervix-classif/data/bounding_boxes_299/` so you can copy them from there to your folder, otherwise ImageJ won't detect that some images have already been tagged). You move between pictures with `a` and `s`, forwards and backwards respectively. Note that you must press one of these keys to activate the macro (when you first open an image, the bounding box won't be shown since the macro hasn't been activated yet.

When you're done drawing bounding boxes, upload your files to `/mnt/nfs/kaggle/cervix-classif/data/bounding_boxes_299/` so they are available for everyone to use.

## Working with images

ImageJ only reads local files, so you really have only two options:
1. Download the dataset onto your local machine and work with that *or*
2. Use `sshfs` to mount the `data` folder to a local folder. This will essentially mount the remote folder into one of your local folders and make life much simpler. This definitely works on linux, I have not tested this on other operating systems.