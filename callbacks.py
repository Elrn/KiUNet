from tensorflow.keras.callbacks import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import utils
import numpy as np
import re
import logging


class monitor(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, dataset=None, fig_size_rate=3):
        super(monitor, self).__init__()
        self.save_dir = save_dir
        self.dataset = dataset.skip(37)
        self.fig_size_rate = fig_size_rate
    #
    def on_epoch_end(self, epoch, logs=None): # 149 ~ 155
        cols, rows = 2, 4
        figure, axs  = plt.subplots(cols, rows, figsize=(rows*3,cols*3))
        figure.suptitle(f'Epoch: "{epoch}"', fontsize=10)
        figure.tight_layout()
        for vol, seg in self.dataset.take(1):
            pred = self.model(vol)
            vol, seg, pred = np.squeeze(vol), np.squeeze(tf.argmax(seg, -1)), np.squeeze(tf.argmax(pred, -1))
        for c in range(cols):
            for r in range(rows):
                axs[c][r].set_xticks([])
                axs[c][r].set_yticks([])
                if c == 0:
                    axs[c][r].imshow(vol[r], cmap='gray')
                    axs[c][r].imshow(seg[r], cmap='Greens', alpha=0.5)
                else:
                    axs[c][r].imshow(vol[r], cmap='gray')
                    axs[c][r].imshow(pred[r], cmap='Reds', alpha=0.5)

        save_path = os.path.join(self.save_dir, f'{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close('all')

    def get_cmap(self):
        transparent = matplotlib.colors.colorConverter.to_rgba('white', alpha=0)
        white = matplotlib.colors.colorConverter.to_rgba('y', alpha=0.5)
        red = matplotlib.colors.colorConverter.to_rgba('r', alpha=0.7)
        return matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [transparent, white, red], 256)
#

class continue_training(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(continue_training, self).__init__()
        self.filepath = os.fspath(filepath) if isinstance(filepath, os.PathLike) else filepath

    def on_train_begin(self, logs=None):
        filepath_to_load = (self._get_most_recently_modified_file_matching_pattern(self.filepath))
        if (filepath_to_load is not None and self._checkpoint_exists(filepath_to_load)):
            try:
                self.model.load_weights(filepath_to_load)
                print(f'[!] Saved Check point is restored from "{filepath_to_load}".')
            except (IOError, ValueError) as e:
                raise ValueError(f'Error loading file from {filepath_to_load}. Reason: {e}')
    #
    def _checkpoint_exists(self, filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        if filepath.endswith('.h5'):
            return tf.io.gfile.exists(filepath)
        tf_saved_model_exists = tf.io.gfile.exists(filepath)
        tf_weights_only_checkpoint_exists = tf.io.gfile.exists(
            filepath + '.index')
        return tf_saved_model_exists or tf_weights_only_checkpoint_exists
    #
    def _get_most_recently_modified_file_matching_pattern(self, pattern):
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

# plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# early_stopping = EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
#     baseline=None, restore_best_weights=False
# )
# ckpt = ModelCheckpoint(
#     filepath, monitor='val_loss', verbose=0, save_best_only=False,
#     save_weights_only=False, mode='auto', save_freq='epoch',
# )

