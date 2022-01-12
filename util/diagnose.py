from __future__ import print_function
import sys
import tensorflow as tf
from tensorflow.contrib import slim


class GraphPrinterHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)


class ArgumentPrinterHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        args = sys.argv[1:]
        print("Arguments:")
        for value in args:
            if value.startswith("-"):
                print("\n%-15s:" % value, end=" ")
            else:
                print(value, end=" ")
        print("\n")
