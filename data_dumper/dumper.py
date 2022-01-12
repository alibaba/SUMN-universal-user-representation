# -*- coding: utf-8 -*-#
import tensorflow as tf


def get_odps_writer(table_name, slice_id):
    return tf.python_io.TableWriter(table_name, slice_id=slice_id)


def get_file_writer(file_path):
    return open(file_path, "wt")