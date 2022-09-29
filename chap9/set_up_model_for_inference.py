# This code is based on demos from https://github.com/facebookresearch/synsin

import torch
import torch.nn as nn

import sys
sys.path.insert(0, './synsin')

import os
os.environ['DEBUG'] = '0'
from synsin.models.networks.sync_batchnorm import convert_model
from synsin.models.base_model import BaseModel
from synsin.options.options import get_model

def synsin_model(model_path):
    '''
    Set up SynSin model:

    Input:
        model_path: path to pretrained model

    Returns:
        Ready model with uploaded weights
    '''

    torch.backends.cudnn.enabled = True

    opts = torch.load(model_path)['opts']
    opts.render_ids = [1]

    model = get_model(opts)

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]

    if 'sync' in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices[0:1]).cuda()
    else:
        model = nn.DataParallel(model, torch_devices[0:1]).cuda()

    #  Load the original model to be tested
    model_to_test = BaseModel(model, opts)
    model_to_test.load_state_dict(torch.load(model_path)['state_dict'])
    model_to_test.eval()

    print("Loaded model")

    return model_to_test