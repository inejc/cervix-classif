function load_roi() {
  name =  getInfo("image.filename");
  name = replace(name, ".jpg", ".roi");
  path = getInfo("image.directory");
  // path_arr = split(path, "/");
  // roi_path = path + name;
  roi_path = "/home/pavlin/kaggle/bounding_boxes/" + name;

  // see if it can be opened
  if (File.exists(roi_path)) {
    open(roi_path);
  } else {
    // print("There is no roi");
    run("Select None");
  }
}

function save_roi_if_exists() {
  name = getInfo("image.filename");
  name = replace(name, ".jpg", ".roi");
  path = getInfo("image.directory");
  // roi_path = path + name;
  roi_path = "/home/pavlin/kaggle/bounding_boxes/" + name;
  if (selectionType() >= 0) {
    // print("Saving to  " + roi_path);
    saveAs("Selection", roi_path);
  } else {
    // print("No roi to save");
  }
}

macro "Prev Image [a]" {
  save_roi_if_exists();
  setKeyDown("alt");
  run("Open Next");
  setKeyDown("none");
  load_roi();
}

macro "Next Image [s]" {
  save_roi_if_exists();
  run("Open Next");
  load_roi();
}
