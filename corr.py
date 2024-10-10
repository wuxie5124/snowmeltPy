# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:49:02 2024

@author: zjm
"""

import numpy as np
import pandas as pd
import sys
import json
import scipy.stats as stats
from minepy import MINE
import warnings
warnings.filterwarnings('ignore')

if __name__  ==  "__main__" :
    a = []
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    excel_path = a[0]
    data=pd.read_excel(excel_path)
    datav = data.values 
    # print(stats.spearmanr(datav, datav)[0][len(datav)-1])
    spearmanr = []
    kendal = []
    mic = []
    mine  = MINE()
    y = datav[:,len(datav[0])-1]
    for i in range(len(datav[0])):
        x = datav[:,i]
        spearmanr.append(stats.spearmanr(x , y)[0])    
        kendal.append(stats.kendalltau(x , y)[0])  
        mine.compute_score(x, y)
        mic.append(mine.mic())
    pearson=data.corr(method="pearson")
    outStr = ""
    value = pearson["Level"].values
    index = pearson.index
    # for i in range(len(value)):
    #     outStr += "#" + index[i] + "&" + str(round(abs(value[i]),2))
    # print(outStr)
    outDict = {}
    for i in range(len(value)):
        coefficient = [];
        coefficient.append(round(abs(value[i]),2))
        coefficient.append(round(abs(spearmanr[i]),2))
        coefficient.append(round(abs(kendal[i]),2))
        coefficient.append(round(abs(mic[i]),2))
        outDict[index[i]] = coefficient
    outStr = json.dumps(outDict);
    print(outStr)
        