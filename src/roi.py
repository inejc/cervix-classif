import struct
import numpy as np
from os.path import join

ROI_TYPE_RECT = 1

def write_bytes(f, n_bytes, what):
    f.write(what.to_bytes(n_bytes, byteorder='big', signed=False))

def write8(f, int8):
    write_bytes(f, 1, int8)

def write16(f, int16):
    write_bytes(f, 2, int16)

def write32(f, int32):
    write_bytes(f, 4, int32)

def writefloat(f, b):
    f.write(struct.pack('f', b))

def write_bb(f, bb):
    """ Save bounding box bb passed in the same format as it is returned from
    ijroi.read_roi into fileobj.
    
    Example usage:
      with open('output.roi', 'wb+' ) as f:
          write_bb(f, np.array([[0, 0], [0, 10], [20, 10], [20, 0]]))
    """

    top,left     = tuple(map(int, bb[0]))
    bottom,right = tuple(map(int, bb[2]))

    f.write(b'Iout')          # magic
    write16(f, 227)           # version
    write8(f, ROI_TYPE_RECT)  # type
    write8(f, 0)              # trash
    write16(f, top)           # top
    write16(f, left)          # left
    write16(f, bottom)        # bottom
    write16(f, right)         # right
    write16(f, 0)             # n_coordinates
    writefloat(f, 0.0)        # x1
    writefloat(f, 0.0)        # y1
    writefloat(f, 0.0)        # x2
    writefloat(f, 0.0)        # y2
    write16(f, 0)             # stroke_width
    write32(f, 0)             # shape_roi_size
    write32(f, 0)             # stroke_color
    write32(f, 0)             # fill_color
    write16(f, 0)             # subtype
    write16(f, 0)             # options
    write8(f, 0)              # arrow_style
    write8(f, 0)              # arrow_head_size
    write16(f, 0)             # rect_arc_size
    write32(f, 0)             # position
    write32(f, 64)            # header2offset
    # we also need fill the file up with 64 zeros
    for i in range(16):
        write32(f, 0)


def save_prediction(array, fname):
    """Write an array of standard bounding box format to a roi file.
    
    Parameters
    ----------
    array : [x, y, w, h]
    fname : string
    
    """
    x, y, w, h = array
    with open(fname, 'wb+') as fhandle:
        write_bb(fhandle, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def save_predictions(img_ids, predictions, output_dir):
    """Dump predictions to an output dir in roi format.
    
    The id positions must correspond with the prediction indices.
    
    """
    for img_id, pred in zip(img_ids, predictions):
        save_prediction(pred, join(output_dir, '%s.roi' % img_id))


def downsize_bb(bb, original_dims, new_dims):
    """ Take bounding box bb as returned from ijroi.read_roi that was created
    over image with (width, height)=original_dims and resize and crop it to
    (width', height')=new_dims """

    bb_c = bb.copy()
    cropped_height = 0
    cropped_width  = 0
    if original_dims[0] > original_dims[1]:
        cropped_width = original_dims[0] - original_dims[1]
    else:
        cropped_height = original_dims[1] - original_dims[0]
    bb_c[:, 0] = (bb[:, 0]-cropped_height/2) / (original_dims[1]-cropped_height) * new_dims[1]
    bb_c[:, 1] = (bb[:, 1]-cropped_width/2) / (original_dims[0]-cropped_width)  * new_dims[0]
    bb_c[:, 0] = np.clip(bb_c[:, 0], 0, new_dims[1])
    bb_c[:, 1] = np.clip(bb_c[:, 1], 0, new_dims[0])
    return bb_c

def get_img_dims(filepath):
    " Return image dimensions (width, height) for image at filepath. "
    im = Image.open(filepath)
    return im.size


if __name__ == '__main__':
    import sys
    from PIL import Image
    import ijroi

    TARGET_SIZE = (299, 299)

    if len(sys.argv) != 3:
        print("Take a list of .roi files that were created over the original (non-resized non-cropped) dataset and fix them to fit our 299x299 dataset format.")
        print("Example usage:")
        print(" $ ls path/to/large/.rois | python3 roi.py path/to/folder/with/images target/path")
        sys.exit()
    rois_to_fix = [roi.strip() for roi in sys.stdin]
    img_dir, output_dir = sys.argv[1:]

    for roi in rois_to_fix:
        base_name = roi[roi.rfind('/')+1:-len('.roi')]
        with open(roi, "rb" ) as f:
            bb = ijroi.read_roi(f)
        original_dims = get_img_dims(img_dir + '/' + base_name + '.jpg')
        bb_resized = downsize_bb(bb, original_dims, TARGET_SIZE)
        with open(output_dir + '/' + base_name + '.roi', "wb+" ) as f:
            write_bb(f, bb_resized)
