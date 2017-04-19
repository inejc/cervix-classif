// Bounding box directory where all the bounding boxes will be placed. Make
// sure to remove trailing slash.
var BOUNDING_BOX_DIR = "bounding_boxes";

function get_roi_path_keep_dir() {
  // Keep the directory structure when saving ROI files, but place them into
  // the specified bounding box directory.
  name =  getInfo("image.filename");
  name = replace(name, ".jpg", ".roi");
  path = getInfo("image.directory");
  path = replace(path, "data", "data/" + BOUNDING_BOX_DIR);
  return path + name;
}

function get_roi_path() {
  // Place all the ROI files into a single directory.
  name =  getInfo("image.filename");
  name = replace(name, ".jpg", ".roi");
  path = getInfo("image.directory");
  path = replace(path, "data/.*", "data/" + BOUNDING_BOX_DIR + "/");
  return path + name;
}

function load_roi() {
  roi_path = get_roi_path();

  // see if it can be opened
  if (File.exists(roi_path)) {
    open(roi_path);
  } else {
    run("Select None");
  }
}

function save_roi_if_exists() {
  roi_path = get_roi_path();

  if (selectionType() >= 0) {
    saveAs("Selection", roi_path);
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
