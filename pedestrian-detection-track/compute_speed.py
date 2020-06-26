#!/usr/bin/env python

import time

import torch
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_speed(model, input_size, device, iteration):
#     device = torch.device("cpu")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    
    # model = model.to(device)
    model = model.to(device)
    model=model.cuda()
    # print(device ,model.type)
    model.eval()

    inputt = torch.randn(*input_size, device=device)#.cuda()
    # input=input.to(device)
    inputt=inputt.cuda()
    #print(inputt)

    for _ in range(10):
        model(inputt.float().cuda())

    logger.info('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(inputt)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    logger.info(
        'Elapsed time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    logger.info('Speed Time: %.2f ms / iter    FPS: %.2f' % (
        elapsed_time / iteration * 1000, iteration / elapsed_time))
