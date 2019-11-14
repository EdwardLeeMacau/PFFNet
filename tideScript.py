import argparse
import os
import pandas as pd

def traverseLog(args):
    tags = os.listdir(args.log)
    stat = {}

    for tag in tags:
        fpath = os.path.join(args.log, tag, 'statistical.xlsx')
        if os.path.exists(fpath):
            df = pd.read_excel(fpath)
            print(fpath, df.shape)
            stat[tag] = df

    return stat 

def traverseOutput(args):
    tags = os.listdir(args.dir)
    stat = {}

    for tag in tags:
        epochs = os.listdir(os.path.join(args.dir, tag))
        for epoch in epochs:
            fpath = os.path.join(os.path.join(args.dir, tag, epoch, './record.xlsx'))
            if os.path.exists(fpath):
                df = pd.read_excel(fpath)
                print(fpath, df)
                stat[(tag, epoch)] = df

    return stat

def main(args):
    traverseLog(args)
    traverseOutput(args)
    return

if __name__ == "__main__":
    args = argparse.ArgumentParser(prog="Experiment Tider")
    args.add_argument("-d", "--dir", required=True, default="./output")
    args.add_argument("-l", "--log", required=True, default="./log")
    
    main(args)