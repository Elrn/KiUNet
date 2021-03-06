import os, datetime, re, logging
import matplotlib.pyplot as plt
import numpy as np
# import nibabel as nib
import SimpleITK as sitk
import tensorflow as tf

########################################################################################################################
""" LOGGING """
# logging.debug(f'') logging.info(f'') logging.warning(f'') logging.error(f'')
format = '[%(asctime)s]|[%(name)s]|[%(levelname)s] %(message)s'
formatter = logging.Formatter(format)
logging.basicConfig(
    # filename='log.txt',
    format=format,
    datefmt='%m/%d/%Y %I:%M:%S',
    level=logging.DEBUG
)
def get_logger(name=None, level=None):
    level = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL][level]
    logger = logging.getLogger(f"{name}")
    if level is not None: logger.setLevel(level)

    return logger

# stream_hander = logging.StreamHandler()
# stream_hander.setFormatter(formatter)
# logger.addHandler(stream_hander)


########################################################################################################################
safe_divide = lambda a, b : np.divide(a, b, out=np.zeros_like(a), where=b!=0)
to_list = lambda x: [x] if type(x) is not list else x

########################################################################################################################
def checkpoint_exists(filepath):
    """Returns whether the checkpoint `filepath` refers to exists."""
    if filepath.endswith('.h5'):
        return tf.io.gfile.exists(filepath)
    tf_saved_model_exists = tf.io.gfile.exists(filepath)
    tf_weights_only_checkpoint_exists = tf.io.gfile.exists(
        filepath + '.index')
    return tf_saved_model_exists or tf_weights_only_checkpoint_exists

def get_checkpoint(filepath, epoch=None):
    # filesystem
    filepath = os.fspath(filepath) if isinstance(filepath, os.PathLike) else filepath
    def _get_most_recently_modified_file_matching_pattern(pattern):
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

        latest_tf_checkpoint = tf.train.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(
                base_name_regex, os.path.basename(latest_tf_checkpoint)):
            return latest_tf_checkpoint

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if tf.io.gfile.exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (file_path_with_largest_file_name is None or
                            # epoch??? ??? ??????????????? last saved model??? ??????
                            file_path > file_path_with_largest_file_name):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found, reset
                        # the counter for the number of files with latest modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the most recent,
                        # increment the counter for the number of files with latest modified
                        # time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time, return
            # the file path with the largest file name.
            return file_path_with_largest_file_name

    return _get_most_recently_modified_file_matching_pattern(filepath)

########################################################################################################################
def join_dir(dirs:list):
    if len(dirs) == 0:
        return dirs
    base = dirs[0]
    for dir in dirs[1:]:
        base = os.path.join(base, dir)
    return base
#
def mkdir(path):
    try: # if hasattr(path, '__len__') and type(path) != str:
        os.makedirs(path)
    except OSError as error:
        print(error)
#

# date = datetime.datetime.today().strftime('%Y-%m-%d_%Hh%Mm%Ss')

#
def save_history(history, path):
    metrics = list(history.history)
    len_metrics = int(len(metrics) // 2)
    fig, ax = plt.subplots(1, len_metrics, )
    # ax = ax.ravel()  # flatten
    for i in range(len_metrics):
        ax[i].plot(history.history[metrics[i]])
        ax[i].plot(history.history["val_" + metrics[i]])
        ax[i].set_title(f"Model {metrics[i]}")
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metrics[i])
        ax[i].legend(["train", "val"])
    fig.tight_layout()  # fix overlapping between title and labels
    fig.savefig(f'{path}', dpi=300)

########################################################################################################################
# DataSet Utils
"""
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
"""
########################################################################################################################
def SWN(image, level=50, window=250, dtype='float32'):
    """ stochastic tissue window normalization
    https://arxiv.org/abs/1912.00420

    # value = {'brain':        [40, 80], * 10?!
    #          'lungs':        [-600, 1500],
    #          'liver':        [30, 150],
    #          'Soft tissues': [50, 250],
    #          'bone':         [400, 1800],
    #          }
    """
    def preprocessing(x):
        if type(x) == list or type(x) == tuple:
            return x[0], x[1]
        elif type(x) == int:
            return x, 0
        else:
            raise TypeError(f'Input type must "list", "tuple" or "int" but "{type(x)}".')

    image = tf.cast(image, dtype)
    level_mean, level_std = preprocessing(level)
    window_mean, window_std = preprocessing(window)

    level = tf.random.normal([1], level_mean, level_std)
    window = tf.random.normal([1], window_mean, window_std)
    window = tf.math.abs(window)

    max_threshold = level + window / 2
    min_threshold = level - window / 2

    image = tf.clip_by_value(image, min_threshold, max_threshold)
    (image - min_threshold) / (max_threshold - min_threshold)
    return image


def read_medical_file(file_path, ):
    # _, ext = os.path.splitext(file_path)
    print(os.path.basename(file_path))
    image = sitk.ReadImage(file_path, sitk.sitkFloat64)  # channel first / i.e sitk.sitkInt16
    ndarray = sitk.GetArrayFromImage(image)
    ndarray = np.transpose(ndarray, [1, 2, 0])  # make channel last
    # if ext == 'dcm':
    #     # ndarray = dcm.read_file(file_path).pixel_array # return non-channel
    #     image = sitk.ReadImage(file_path, sitk.sitkFloat64)  # channel first / i.e sitk.sitkInt16
    #     ndarray = sitk.GetArrayFromImage(image)
    #     ndarray = np.transpose(ndarray, [1, 2, 0])  # make channel last
    # elif ext == 'nii' or 'nii.gz' in os.path.basename(file_path) :
    #     ndarray = nib.load(file_path).get_fdata() # channel last
    # else:
    #     raise ValueError(f"Unexpected Medical file's extension, [{file_path}]")
    if not isinstance(ndarray, np.ndarray):
        raise ValueError(f'Type of Image must "ndarray", but "{type(ndarray)}" has returned.')
    return ndarray


def resampling(image, spacing=None, default_value=0):
    """
    sitk file??? spacing??? ??????, ??????
    image size??? origin??? ??????
    """
    img_sz = list(image.GetSize())

    if type(spacing) == float:
        spacing = [spacing] * len(img_sz)
    if type(spacing) == int:
        spacing = [float(spacing)] * len(img_sz)
    elif type(spacing) == list:
        if len(spacing) != len(img_sz):
            raise ValueError(f'')
    else:
        raise ValueError(f'')

    resize_rate = np.divide(list(image.GetSpacing()), spacing)
    target_sz = np.multiply(img_sz, resize_rate).astype('int32').tolist()

    print(f'[!] Original Image size "{img_sz}" resized to "{target_sz}".')

    def get_origin(image, spacing, size):
        sz = np.array(image.GetSize())
        sp = np.array(image.GetSpacing())
        new_size_no_shift = np.int16(np.ceil(sz * sp / spacing))
        origin = np.array(image.GetOrigin())
        shift_amount = np.int16(np.floor((new_size_no_shift - size) / 2)) * spacing
        return origin + shift_amount

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetDefaultPixelValue(default_value)
    resample.SetOutputSpacing(spacing)
    resample.SetSize(target_sz)
    resample.SetOutputOrigin(get_origin(image, spacing, target_sz))

    return resample.Execute(image)


########################################################################################################################
def none_zeros_coord(image):
    """
    ndarray ??? 0??? ?????? ????????? ???????????? ??????
    """
    mask = image > 0
    return np.argwhere(mask)

def make_paddings(diff):
    diff *= diff > 0
    paddings = np.stack([(diff+1)//2, diff//2], 1)
    assert False not in (np.sum(paddings, -1) == diff)
    return paddings

def crop_voxel(image, tartget_size, crop_ratio=None, use_zero_crop=None):
    """
    crop_ratio = ??? ?????? ?????? crop ?????? ?????? weight ???
    """
    shape = np.array(image.shape)
    tartget_size = np.array(tartget_size)
    assert len(shape) == len(tartget_size)
    assert len(tartget_size) == 3
    need_crop = (image.shape - tartget_size)

    def solve_around_problem(ratio):
        """
        np ???????????? .5 ??? ?????????. ????????? ????????? ????????? .5??? ?????????????????? ????????? ????????? ????????? ?????????.
        """
        ratio = np.transpose(ratio)
        ratio[0] = np.ceil(ratio[0])
        ratio[1] = np.floor(ratio[1])
        ratio = np.transpose(ratio)
        return ratio

    def index_translate(ratio, img_shape):
        """
        crop??? ?????? indexing ?????? ??????,
        -0 ??? ????????? size??? ??????
        """
        ratio = np.transpose(ratio)
        ratio[1] *= -1
        for i, value in enumerate(ratio[1]):
            if value == 0:
                ratio[1][i] = img_shape[i]
        return ratio

    #### Crop zero padding Block ####
    if use_zero_crop == True:
        coords = none_zeros_coord(image)
        check_margin = np.stack([coords.min(0), shape - (coords.max(0) + 1)], 1)  # ????????? ??????
        total_margins = np.sum(check_margin, 1)
        margin_ratio = np.divide(check_margin, np.stack([total_margins, total_margins], 1))
        margin_ratio = np.nan_to_num(margin_ratio)
        crop_margin = total_margins - need_crop
        zero_crop = need_crop * (crop_margin > 0)
        need_crop -= zero_crop
        amount_margin_crop = margin_ratio * np.stack([zero_crop, zero_crop], 1)
        amount_margin_crop = solve_around_problem(amount_margin_crop).astype(np.int32)
        index_translate(amount_margin_crop, image.shape)

        image = image[
                amount_margin_crop[0][0]:amount_margin_crop[0][1],
                amount_margin_crop[1][0]:amount_margin_crop[1][1],
                amount_margin_crop[2][0]:amount_margin_crop[2][1]]

    #### Crop Image Block ####
    amount_image_crop = need_crop
    if crop_ratio is not None:
        # ?????? [1, 0] ?????? ????????? ?????? image ??? ?????? broadcasting
        if len(np.array(crop_ratio)) == 2:
            crop_ratio = np.tile(crop_ratio, [len(shape), 1])
        if np.any(np.sum(crop_ratio, 1) > 1):
            raise ValueError(f"crop_ratio's sum MUST under '1', but got {np.sum(crop_ratio, 1)}")
        assert len(shape) == len(np.array(crop_ratio))
        amount_image_crop = np.stack([amount_image_crop, amount_image_crop], 1).astype(float)
        amount_image_crop *= crop_ratio
        amount_image_crop = solve_around_problem(amount_image_crop)
    else:
        amount_image_crop = np.stack([amount_image_crop + 1, amount_image_crop], 1) // 2
    amount_image_crop = amount_image_crop.astype(np.int32)
    index_translate(amount_image_crop, image.shape)

    image = image[
            amount_image_crop[0][0]:amount_image_crop[0][1],
            amount_image_crop[1][0]:amount_image_crop[1][1],
            amount_image_crop[2][0]:amount_image_crop[2][1]]

    print(f'[!] Image Cropped "{need_crop}", Original size "{shape.tolist()}" to "{image.shape}".')
    return image

def crop_n_pad(image, size, axis=None, crop_ratio=None, use_zero_crop=None):
    img_shape = np.array(image.shape)
    size = list(size)
    if axis == None:
        # size ??? ????????? ?????? image rank ??? ?????? size rank ??? ?????? size??? ?????? ????????? ??????
        if len(size) == 1 and len(size) != len(img_shape):
            size *= len(img_shape)
        if len(img_shape) != len(size):
            raise ValueError(f'')
    else:
        # size rank??? axis rank??? ???????????? ??????, size??? image rank??? ?????? ??????
        axis = np.array(axis)
        if len(np.array(size)) != len(axis):
            raise ValueError(f'')
        if len(axis) != 3:
            axis_to_dim = {ax: size[i] for i, ax in enumerate(axis)}
            size = [axis_to_dim[i] if i in axis else img_shape[i] for i in range(len(img_shape))]

    # [Padding Block]
    paddings = make_paddings(size - img_shape)
    padded_image = np.pad(image, paddings) #
    print(f'[!] Image Padded "{paddings.tolist()}", Original size "{image.shape}" to "{padded_image.shape}"')

    # [Cropping Block]
    # arr??? zero ??? ??????????????? ????????? ?????? ??????.
    # zero ????????? ?????? ???????????? ?????? ?????? ?????? crop ??? ?????? crop??? ??????.
    if use_zero_crop == True:
        padded_image = crop_voxel(padded_image, size, crop_ratio, use_zero_crop=use_zero_crop)
    else:
        padded_image = crop_voxel(padded_image, size, crop_ratio)

    return padded_image


def crop_zeros(image):
    rank = len(image.shape)
    if rank != 2 and rank != 3:
        raise ValueError(f'Image Rank must 2 or 3, but {rank}.')
    # 0??? ?????? ????????? ????????? ???
    coords = none_zeros_coord(image)
    min, max = coords.min(0), coords.max(0) + 1

    image = image[min[0]:max[0], min[1]:max[1]]
    if rank == 3:
        image = image[:, :, min[2]: max[2]]
    return image


def count_patches(input_shape, sizes, strides, paddings='VALID'):
    """
    : Need to reflect variables to number of patches along paddings
        XY(Z) ????????? ??????,
    extract patches ??? ?????? ?????? Convolution ??? output image size ??????
    """
    assert len(sizes)==len(strides)
    to_arr = lambda x:np.array(x)
    shape = (to_arr(input_shape) - to_arr(sizes)) / to_arr(strides) + 1
    return shape.astype(int)

def reconstruct_pathces(input, batch_size, origin_size, ksize, strides):
    """
    : patch size ??? strides ??? ???????????? ?????? ??? reconstruction ??? ???????????? ??????.
        multi-channel ?????? ??????!
    """
    # except batch/channel dimension
    num_patches_along_dims = count_patches(origin_size, ksize, strides)[1:-1]
    # batch ??????????????? ?????? batch dimension??? ??????
    input = tf.stack(tf.split(input, batch_size, 0), 0)
    # input = tf.reshape(input, [batch_size, x.shape[0]//batch_size]+input.shape[1:]])
    split_dim = 1
    for axis, num_patches in enumerate(reversed(num_patches_along_dims)):
        if num_patches == 1: continue # patch ??? 1????????? concat ??? ????????? ??????
        split = tf.split(input, input.shape[split_dim]//num_patches, split_dim)
        # axis ??? extract patches ??? extract dimension ????????? ?????? ??????
        concat_dim = -(axis+2) # Z > Y > X
        split = [tf.concat(tf.split(x, x.shape[split_dim], split_dim), concat_dim) for x in split]
        """ for ?????? ???????????? ?????? ?????? 
        ??? ?????? ?????? ?????? ??? split ?????? ????????? ????????? loop ??? ??????????????????,
        ?????? ?????? ????????? batch dimension ??? ????????? patch ?????? ????????? ????????? 
        split ?????? ????????? list ??? ????????? ?????? patch ?????? ?????? batch dimension ?????? ??????
        """
        perm = [i for i in range(tf.rank(split))][3:]
        input = tf.squeeze(tf.transpose(split, [1,2,0]+perm), split_dim)
    # split dimension ??? ??????
    output = tf.squeeze(input, axis=split_dim)
    return output