import torch
import util, train

train_on_gpu = torch.cuda.is_available()


def train():
    
