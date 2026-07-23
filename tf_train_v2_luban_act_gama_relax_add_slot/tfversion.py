import utils as ut
import argparse
import copy
import sys, os
import numpy as np
import tensorflow as tf
import datetime
from datetime import timedelta
import model_conf
from model import Model
from tensorflow.keras import regularizers
import time

print("version: ", tf.__version__)
