"""
  FileName     [ process.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Draw the statistics message of the training record ]
"""

import ast
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

with open("statistics.txt", "r") as textfile:
    lines = textfile.readlines()
    
    trainloss = ast.literal_eval(lines[0])
    test_mse  = ast.literal_eval(lines[1])
    test_psnr = ast.literal_eval(lines[2])
    test_ssim = ast.literal_eval(lines[3])
    avg_mse   = np.average(test_mse, axis=1)
    avg_psnr  = np.average(test_psnr, axis=1)
    avg_ssim  = np.average(test_ssim, axis=1)
    std_mse   = np.std(test_mse, axis=1)
    std_psnr  = np.std(test_psnr, axis=1)
    std_ssim  = np.std(test_ssim, axis=1)
    epochs    = ast.literal_eval(lines[4])

    # Save the psnr, avg_psnr, std_psnr out
    # Save the ssim, avg_ssim, std_ssim out
    df = pd.DataFrame(np.transpose(test_psnr))
    df = df.append(pd.DataFrame([np.transpose(avg_psnr), np.transpose(std_psnr)]), ignore_index=True)
    df = df.append(pd.DataFrame(np.transpose(test_ssim)), ignore_index=True)
    df = df.append(pd.DataFrame([np.transpose(avg_ssim), np.transpose(std_ssim)]), ignore_index=True)
    # print(df.head)
    df.to_csv("stat_cal.csv")
        
    
    # Plot the graphs, global setting
    plt.figure(figsize=(12.8, 7.2))
    train_val_loss = "train_val_loss.png"
    avg_psnr_ssim  = "avg_psnr_ssim.png"
    detail_psnr    = "detail_PSNR.png"
    detail_ssim    = "detail_SSIM.png"

    # Plot train_loss and test_mse
    """
    plt.clf()
    plt.plot(epochs[1:], trainloss, label="TrainLoss", color='b')
    plt.plot(epochs, avg_mse, label="ValLoss", color='r')
    plt.plot(epochs, np.repeat(np.amin(avg_mse), len(epochs)), ':')
        
    plt.legend(loc=0)
    plt.xlabel("Epoch(s)")
    plt.title("Loss vs Epochs")
    plt.savefig(train_val_loss)
    """

    # Plot PSNR and SSIM
    """
    plt.clf()
    fig, axis1 = plt.subplots(sharex=True)
    axis1.set_xlabel('Epoch(s)')
    axis1.set_ylabel('Average PSNR')
    axis1.plot(epochs, avg_psnr, label="PSNR vs Epochs", color='b')
    axis1.plot(epochs, np.repeat(np.amax(avg_psnr), len(epochs)), ':')
    axis1.tick_params()
    
    axis2 = axis1.twinx()
    axis2.plot(epochs, avg_ssim, label="SSIM vs Epochs", color='r')
    axis2.set_ylabel('Average SSIM')
    axis2.tick_params()
        
    plt.legend(loc=0)
    plt.title("PSNR-SSIM vs Epochs")
    plt.savefig(average_psnr_ssim)
    """    

    # Plot Mean and Std of the PSNR / SSIM
    """
    plt.clf()
    fig, axis1 = plt.subplots(sharex=True)
    axis1.set_xlabel('Epoch(s)')
    axis1.set_ylabel('Average PSNR')
    axis1.plot(epochs, avg_psnr, label="PSNR vs Epochs", color='b')
    axis1.plot(epochs, np.repeat(np.amax(avg_psnr), len(epochs)), ':')
    axis1.tick_params()
    
    axis2 = axis1.twinx()
    axis2.plot(epochs, avg_ssim, label="SSIM vs Epochs", color='r')
    axis2.set_ylabel('Average SSIM')
    axis2.tick_params()
        
    plt.legend(loc=0)
    plt.title("PSNR-SSIM vs Epochs")
    plt.savefig(detail_ssim)
    """

    # Plot the detail of the PSNR
    """
    plt.clf()
    
    test_psnr = np.transpose(test_psnr)
    for i in range(0, 5):
        plt.plot(epochs, test_psnr[i], label="TestLoss_{} vs Epochs".format(31+i))
        
    plt.legend(loc=0)
    plt.title("PSNR vs Epochs")
    plt.savefig(detail_psnr)
    """

    # Plot the detail of the SSIM
    """
    plt.clf()
    
    test_psnr = np.transpose(test_psnr)
    for i in range(0, 5):
        plt.plot(epochs, test_psnr[i], label="TestLoss_{} vs Epochs".format(31+i))
        
    plt.legend(loc=0)
    plt.title("PSNR vs Epochs")
    plt.savefig(detail_ssim)
    """