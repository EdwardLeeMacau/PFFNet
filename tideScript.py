import argparse
import os
import pandas as pd

import utils

def traverseLog(logpath, number):
    tags = os.listdir(logpath)
    stat = {}

    for tag in tags:
        fpath = os.path.join(logpath, tag, 'statistical.xlsx')
    
        if os.path.exists(fpath):
            df = pd.read_excel(fpath).drop_duplicates().astype(
                {'Iterations': int}
            ).set_index('Iterations').drop(
                axis='columns', 
                labels='Unnamed: 0'
            )
            stat[tag] = df.nsmallest(number, 'ValidationLoss', 'first')

    if len(stat) > 0: 
        stat = pd.concat(stat.values(), axis=0, keys=stat.keys(), names=['Tags'])
    else: 
        stat = pd.DataFrame()

    return stat 

def traverseOutput(outputpath):
    tags = os.listdir(outputpath)
    stat = {}

    for tag in tags:
        epochs = os.listdir(os.path.join(outputpath, tag))
        
        tmp = {}
        for epoch in epochs:
            fpath = os.path.join(os.path.join(outputpath, tag, epoch, './record.xlsx'))

            if os.path.exists(fpath):
                df = pd.read_excel(fpath).rename(
                    columns={'Unnamed: 0': 'Standard'}
                ).set_index('Standard')
                tmp[epoch] = df

        if len(tmp) > 0:
            stat[tag] = pd.concat(tmp.values(), axis=0, keys=tmp.keys(), names=['Iterations'])

    if len(stat) > 0:
        stat = pd.concat(stat.values(), axis=0, keys=stat.keys(), names=['Tags'])
    else:
        stat = pd.DataFrame()
    
    return stat

def main(args):
    trainCurve  = traverseLog(args.log, args.number)
    trainCurve.columns = pd.MultiIndex.from_product([['Train'], trainCurve.columns])

    performance = traverseOutput(args.dir).unstack().swaplevel(i=0, j=1, axis='columns').sort_index(1)
    
    if args.output is not None:
        with pd.ExcelWriter(args.output) as writer:
            trainCurve.to_excel(writer, sheet_name='Train')
            performance.to_excel(writer, sheet_name='Validation')

    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog="Experiment Tider")
    argparser.add_argument("-d", "--dir", default="./output")
    argparser.add_argument("-l", "--log", default="./log")
    argparser.add_argument("-o", "--output")
    argparser.add_argument("-n", "--number", type=int, default=3)

    args = argparser.parse_args()
    
    main(args)
