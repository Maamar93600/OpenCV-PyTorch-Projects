# =========================
# Python standard library
# =========================
import os
import gc
import random
import shutil
from dataclasses import dataclass
from glob import glob


# =========================
# Data / scientific computing
# =========================
import numpy as np
import pandas as pd


# =========================
# Image processing / visualization
# =========================
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# =========================
# Data augmentation
# =========================
import albumentations as A


# =========================
# PyTorch
# =========================
import torch
import torch.amp as amp
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split


# =========================
# PyTorch ecosystem
# =========================
import segmentation_models_pytorch as smp
from torchmetrics import MeanMetric
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


# =========================
# Utilities
# =========================
from tqdm import tqdm


# =========================
# Display
# =========================
bold = "\033[1m"
reset = "\033[0m"