"""
Author: Yanting Miao
"""
import time
import torch
from Model import Transformer
from Optim import TransformerOptim

def timeSince(start):
    end = time.time()