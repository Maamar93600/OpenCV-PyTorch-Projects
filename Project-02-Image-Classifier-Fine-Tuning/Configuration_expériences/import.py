import os
import glob
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from PIL import Image
from dataclasses import dataclass
from itertools import product

from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms as T
from torch.optim import lr_scheduler

from torchinfo import summary

from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix
)

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sklearn.metrics import classification_report
from tqdm import tqdm


bold = f"\033[1m"
reset = f"\033[0m"