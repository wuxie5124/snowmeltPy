# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator
import sys
import os

def get_stacking(clf, x_train, y_train, x_test, n_folds= 4):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        a =  clf.predict(x_tst)
        second_level_train_set[test_index] = a
        test_nfolds_sets[:,i] = clf.predict(x_test)
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

def getMLModule(machinestr):
    MLarray = machinestr.split("%")
    count = len(MLarray)
    name = MLarray[0]
    if(name == 'XGBoost'):
        model = xgb.XGBRegressor()
        for i in range(1,count):
            param = MLarray[i].split('&')[0]
            value = MLarray[i].split('&')[1]
            if(value == "Default"):
                continue;
            elif(param == "learning_rate"):
                model.learning_rate = float(value)
            elif(param == "n_estimators"):
                model.n_estimators = int(value)
            elif(param == "max_depth"):
                model.max_depth = int(value)
            elif(param == "min_child_weight"):
                model.min_child_weight = int(value)
            elif(param == "colsample_bytree"):
                model.colsample_bytree = float(value)
            elif(param == "reg_alpha"):
                model.reg_alpha =float(value)
            elif(param == "reg_lambda"):
                model.reg_lambda = int(value)
             
    elif(name == 'GBDT'):
        model = GradientBoostingRegressor()
        for i in range(1,count):
            param = MLarray[i].split('&')[0]
            value = MLarray[i].split('&')[1]
            if(value == "Default"):
                continue;
            elif(param == "learning_rate"):
                model.learning_rate = float(value)                
            elif(param == "n_estimators"):
                model.n_estimators = int(value)
            elif(param == "min_samples_split"):
                model.min_samples_split = int(value)
            elif(param == "min_samples_leaf"):
                model.min_samples_leaf = int(value)
            elif(param == "max_features"):
                model.max_features = value    
            elif(param == "subsample"):
                model.subsample = float(value)
            elif(param == "random_state"):
                model.random_state = int(value)
    elif(name == 'KNN'):
        model = KNeighborsRegressor()
        for i in range(1,count):
            param = MLarray[i].split('&')[0]
            value = MLarray[i].split('&')[1]
            if(value == "Default"):
                continue;
            elif(param == "n_neighbors"):
                model.learning_rate = float(value)                
            elif(param == "weights"):  
                model.weights = value
    elif(name == 'RandomForest'):
        model = RandomForestRegressor()
        for i in range(1,count):
            param = MLarray[i].split('&')[0]
            value = MLarray[i].split('&')[1]
            if(value == "Default"):
                continue;
            elif(param == "n_estimators"):
                model.n_estimators = int(value)     
            elif(param == "min_samples_split"):
                model.min_samples_split = int(value)
            elif(param == "min_samples_leaf"):
                model.min_samples_leaf = int(value)  
            elif(param == "max_depth"):
                model.max_depth = int(value)   
            elif(param == "max_features"):
                model.max_features = value     
            elif(param == "random_state"):
                model.random_state = int(value)                
    elif(name == 'SVM'):
        model = SVR()
        for i in range(1,count):
            param = MLarray[i].split('&')[0]
            value = MLarray[i].split('&')[1]        
            if(value == "Default"):
                continue;            
            elif(param == "C"):
                model.random_state = float(value)   
            elif(param == "kernel"):
                model.kernel = value

    return model       

def loadData(filePath,param):
    if(filePath.split(".")[1] == "xls" or filePath.split(".")[1] == "xlsx"):
        PD = pd.read_excel(filePath)
    elif(filePath.split(".")[1] == "csv"):
        PD = pd.read_csv(filepath_or_buffer = filePath)  
    num = len(PD)
    a = list(range(0,num,1))
    b = list(range(0,num,5))
    for i in b:
        a.remove(i)
    data_train = PD.iloc[a,:]
    data_test  = PD.iloc[b,:]
    data_num1 = len(data_train)
    data_num2 = len(data_test)
    XList_train = []
    XList_test  = []
#    param = PD.columns.values
    for row in range(0, data_num1):
        tmp_list = []
        for low in range(len(param)):
            pn = param[low]
            tmp_list.append(data_train.iloc[row][pn])
        XList_train.append(tmp_list)
    ylist_train =data_train.Level.values
    for row in range(0, data_num2):
        tmp_list = []
        for low in range(len(param)):
            pn = param[low]
            tmp_list.append(data_test.iloc[row][pn])
        XList_test.append(tmp_list)
    ylist_test =data_test.Level.values   
    ylist_lon =data_test.lon.values
    ylist_lat =data_test.lat.values
    
    return np.array(XList_train),np.array(XList_test),ylist_train,ylist_test,ylist_lon,ylist_lat  

def writeTiff(values,sourceTifPath,targertTifPath):
    dataset = gdal.Open(sourceTifPath)   
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
    
    nodataValue = dataset.GetRasterBand(1).GetNoDataValue()
    valuetif = values.reshape(im_height,im_width)
    valuetif[im_data == nodataValue] = nodataValue
    datatype = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")
    rdband = 1
    dataset2 = driver.Create(targertTifPath, im_width, im_height, rdband, datatype)
    # rdgeotrans = (self.west, self.width, 0, self.north, 0, -self.hight)
    dataset2.SetGeoTransform(im_geotrans)
    rdproj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
    dataset2.SetProjection(rdproj)
    band = dataset2.GetRasterBand(1)
    band.SetNoDataValue(nodataValue)
    band.WriteArray(valuetif)
    dataset2 = None
    band = None

def readTif(path):

    dataset = gdal.Open(path)   
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
    return im_data.reshape(-1,).tolist()

def readLonAndLat(path):

    dataset = gdal.Open(path)   
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
    left = round(im_geotrans[0],10)
    top = round(im_geotrans[3],10)
    xcel = round(im_geotrans[1],10)
    ycel = round(im_geotrans[4],10)
    xarr = []
    yarr = [] 
    for i in range(im_height):
        for j in range(im_width):
            lon = left + xcel*j
            lat = top + ycel*i
            xarr.append(lon)
            yarr.append(lat)
    return xarr,yarr

def loadTif(tifPaths):
    xprect = []
    if(len(tifPaths) > 0 ):
        lon,lat = readLonAndLat(tifPaths[0])
        xprect.append(lon)
        xprect.append(lat)
    for i in range(len(tifPaths)):
        data = readTif(tifPaths[i])
        xprect.append(data)
    return np.array(xprect).T    

def calculate(machines, params, tifPaths, excelFilePathStrs):
    ML1 = getMLModule(machines[0])
    ML2 = getMLModule(machines[1])
    ML3 = getMLModule(machines[2])
    ML4 = getMLModule(machines[3])
    ML5 = getMLModule(machines[4])
    ML21 = getMLModule(machines[5])
    
    train_x, test_x, train_y, test_y,lon_y,lat_y = loadData(excelFilePathStrs ,params)
    # train_y = le.fit_transform(train_y)
    loadPredictX = loadTif(tifPaths) 
    train_sets = []
    test_sets = []
    temptifPath = tifPaths[0]
    # print("train_x 形状：" + str(train_x.shape))
    # print("train_y 形状：" + str(train_y.shape))
    # print("loadPredictX 形状：" + str(loadPredictX.shape))
    #散点图
    for clf in [ML1, ML2, ML3, ML4, ML5]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
        train_sets.append(train_set)
        test_sets.append(test_set)   
        test_arr = np.array(test_sets)  
    # print(test_arr.shape)
    # print(test_y.shape)
    for i in range(5):
        sandiantu(test_y,test_sets[i],machines[i].split("%")[0]);     
        
    meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

    ML21.fit(meta_train, train_y)
    df_predict = ML21.predict(meta_test)   
    sandiantu(test_y,df_predict,"Stacking");     
    
    #散点图
    train_sets = []
    test_sets = []
    # 输出结果     
    for clf in [ML1, ML2, ML3, ML4, ML5]:
        train_set, test_set = get_stacking(clf, train_x, train_y, loadPredictX)
        train_sets.append(train_set)
        test_sets.append(test_set)   
        test_arr = np.array(test_sets)  
    # targettifPath = os.path.dirname(excelFilePathStrs) 
    print(directory + "\\" + machines[0].split("%")[0] +".tif")   
    writeTiff(test_arr[0],temptifPath,directory + "\\" + machines[0].split("%")[0] +".tif")
    writeTiff(test_arr[1],temptifPath,directory + "\\" + machines[1].split("%")[0] +".tif")
    writeTiff(test_arr[2],temptifPath,directory + "\\" + machines[2].split("%")[0] +".tif")
    writeTiff(test_arr[3],temptifPath,directory + "\\" + machines[3].split("%")[0] +".tif")
    writeTiff(test_arr[4],temptifPath,directory + "\\" + machines[4].split("%")[0] +".tif")
    # for i in range(5):
    #     sandiantu(test_y,test_sets[i],machines[i].split("%")[0]);
    print("第一层结果:" + directory + "\\" + machines[0].split("%")[0] +".tif")
    print("第一层结果:" + directory + "\\" + machines[1].split("%")[0] +".tif")
    print("第一层结果:" + directory + "\\" + machines[2].split("%")[0] +".tif")
    print("第一层结果:" + directory + "\\" + machines[3].split("%")[0] +".tif")
    print("第一层结果:" + directory + "\\" + machines[4].split("%")[0] +".tif")
    meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

    ML21.fit(meta_train, train_y)
    df_predict = ML21.predict(meta_test)

    writeTiff(df_predict,temptifPath,directory + "\\Stacking.tif")
    # sandiantu(test_y,df_predict,"Stacking");
    print("第二层结果:" + directory + "\\Stacking.tif")
def sandiantu(x,y,name):
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.figure(figsize=(10, 7), dpi=300)
    # my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    font1 = {
    'weight' : 'normal',
    'size'   : 20,
    }
    Xnp = np.array(x)
    Ynp = np.array(y)
    Xmean = np.mean(Xnp) 
    parameter = np.polyfit(Xnp, Ynp, 1)
    Y2 = parameter[0]*Xnp + parameter[1]
    a = (pd.DataFrame(Ynp,columns = ['Y']).Y).corr((pd.DataFrame(Xnp,columns = ['X'])).X)
    x1 = min(Xnp)
    x2 = max(Xnp)+1
    y1 = parameter[0] * x1 + parameter[1]
    y2 = parameter[0] * x2 + parameter[1]
    plt.scatter(Xnp, Ynp,color = 'black',s = 100 , facecolors='none', edgecolors='black')
    plt.plot([x1,x2], [x1,x2], color='black')
    plt.plot([x1,x2], [y1,y2], color='r')   
    MSE = np.round(np.mean(np.square(Xnp - Ynp)),3)
    NSE =  np.round((1-np.sum(np.square(Xnp - Ynp))/np.sum(np.square(Xnp - Xmean))),3)
    RMSE =  np.round(np.sqrt(np.mean(np.square(Xnp - Ynp))),3)
    MAE =  np.round(np.mean(np.abs(Xnp - Ynp)),3)
    ubRMSE =  np.round(np.sqrt(np.mean(np.square(Xnp - Ynp - np.mean(Xnp) + np.mean(Ynp)))),3)
    cc = np.round(np.corrcoef(Xnp, Ynp)[0][1],3)
    r2 = np.round(r2_score(Xnp, Ynp),3)
    Bias = np.round((np.mean(Ynp)/np.mean(Xnp)-1),3)
    plt.xlim(0, x2)
    plt.ylim(0, x2)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.xticks(fontproperties='Times New Roman', size=26)    
    # plt.yticks([0,1,2,3],fontproperties='Times New Roman', size=26)
    #     plt.xticks([0,1,2,3],fontproperties='Times New Roman', size=26)    
#             x2*0.04 , x2*0.68
    plt.text(x2*0.04 , x2*0.52, name+ '\nCC = ' + str(cc) +   '\nBias = ' + str(Bias)  +  '\nRMSE = ' + str(RMSE) + '\nNSE = '+ str(NSE),fontproperties='Times New Roman',size = 28)
    plt.ylabel('Estimated Level',fontproperties='Times New Roman',size = 28)
    plt.xlabel('Real Level',fontproperties='Times New Roman',size = 28)
    plt.savefig(directory + '\\' + name + u"散点图")
    print("成功图保存在:" + directory + '\\' + name + u"散点图")
    # plt.show()

if __name__ == "__main__":
    a = []
    # excelFilePathStrs = r"C:\\Users\\zhangjunmin\\Desktop\\11\\建模参数1.xls"
    # machineStrs = r"#XGBoost%learning_rate&Default%n_estimators&Default%max_depth&Default%min_child_weight&Default%subsample&Default%colsample_bytree&Default%reg_alpha&Default%reg_lambda&Default#GBDT%n_estimators&Default%min_samples_split&Default%min_samples_leaf&Default%max_features&Default%subsample&Default%learning_rate&Default%random_state&Default#KNN%n_neighbors&Default%weights&uniform#RandomForest%n_estimators&Default%min_samples_split&Default%min_samples_leaf&Default%max_depth&Default%max_features&Default%random_state&Default#SVM%C&Default%kernel&rbf#XGBoost%learning_rate&Default%n_estimators&Default%max_depth&Default%min_child_weight&Default%subsample&Default%colsample_bytree&Default%reg_alpha&Default%reg_lambda&Default"
    # tifPathStrs = r"#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Agricultur.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Curve_numb.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Distance_f.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Elevation.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\GDP.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\landuse2000.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Maximum_snow_depth.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Particle_s.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Population.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Relative_E.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Road_densi.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Runoff_CV.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SCDavg.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SCDchangerate.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SCMchangerate5.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SCSchangerate5.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SDSameMonthMean01.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SDSameMonthMean02.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SDSameMonthMean03.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SDSameMonthStd01.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SDSameMonthStd02.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\SDSameMonthStd03.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Slope.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Variance_c.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\Vegetation.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\XJAVHRR_av.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\XJmonthcgrt1.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\XJmonthcgrt2.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\XJmonthcgrt3.tif#%C:\Users\zjm\.spyder-py3\mlearn\xTIFF\577x309\yearchangerate.tif"
    # tifPathStrs = r"#%C:\Users\zhangjunmin\Desktop\tif\Agricultur.tif#%C:\Users\zhangjunmin\Desktop\tif\Curve_numb.tif#%C:\Users\zhangjunmin\Desktop\tif\Distance_f.tif#%C:\Users\zhangjunmin\Desktop\tif\Elevation.tif#%C:\Users\zhangjunmin\Desktop\tif\GDP.tif#%C:\Users\zhangjunmin\Desktop\tif\landuse2000.tif#%C:\Users\zhangjunmin\Desktop\tif\Particle_s.tif#%C:\Users\zhangjunmin\Desktop\tif\Population.tif#%C:\Users\zhangjunmin\Desktop\tif\Relative_E.tif#%C:\Users\zhangjunmin\Desktop\tif\Road_densi.tif#%C:\Users\zhangjunmin\Desktop\tif\Runoff_CV.tif#%C:\Users\zhangjunmin\Desktop\tif\SCDavg.tif#%C:\Users\zhangjunmin\Desktop\tif\SCDchangerate.tif#%C:\Users\zhangjunmin\Desktop\tif\SCMchangerate5.tif#%C:\Users\zhangjunmin\Desktop\tif\SCSchangerate5.tif#%C:\Users\zhangjunmin\Desktop\tif\SDSameMonthMean01.tif#%C:\Users\zhangjunmin\Desktop\tif\SDSameMonthMean02.tif#%C:\Users\zhangjunmin\Desktop\tif\SDSameMonthMean03.tif#%C:\Users\zhangjunmin\Desktop\tif\SDSameMonthStd01.tif#%C:\Users\zhangjunmin\Desktop\tif\SDSameMonthStd02.tif#%C:\Users\zhangjunmin\Desktop\tif\SDSameMonthStd03.tif#%C:\Users\zhangjunmin\Desktop\tif\Slope.tif#%C:\Users\zhangjunmin\Desktop\tif\Variance_c.tif#%C:\Users\zhangjunmin\Desktop\tif\Vegetation.tif#%C:\Users\zhangjunmin\Desktop\tif\XJAVHRR_av.tif#%C:\Users\zhangjunmin\Desktop\tif\XJmonthcgrt1.tif#%C:\Users\zhangjunmin\Desktop\tif\XJmonthcgrt2.tif#%C:\Users\zhangjunmin\Desktop\tif\XJmonthcgrt3.tif#%C:\Users\zhangjunmin\Desktop\tif\yearchangerate.tif"
    
    # # paramStrs = r"#Agricultur#Curve_numb#Distance_f#Elevation#GDP#landuse200#Maximum_sn#Particle_s#Population#Relative_E#Road_densi#Runoff_CV#SCDavg#SCDchanger#SCMchanger#SCSchanger#SDSameMont#SDSameMo_1#SDSameMo_2#SDSameMo_3#SDSameMo_4#SDSameMo_5#Slope#snowclass#Variance_c#Vegetation#XJAVHRR_av#XJmonthcgr#XJmonthc_1#XJmonthc_2#yearchange"    
    # paramStrs = r"#Agricultur#Curve_numb#Distance_f#Elevation#GDP#landuse200#Particle_s#Population#Relative_E#Road_densi#Runoff_CV#SCDavg#SCDchanger#SCMchanger#SCSchanger#SDSameMont#SDSameMo_1#SDSameMo_2#SDSameMo_3#SDSameMo_4#SDSameMo_5#Slope#Variance_c#Vegetation#XJAVHRR_av#XJmonthcgr#XJmonthc_1#XJmonthc_2#yearchange"    
    
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    
    machineStrs = a[0]
    paramStrs =   a[1]
    tifPathStrs = a[2]
    excelFilePathStrs = a[3]
    directory = os.path.dirname(excelFilePathStrs)
    paramStrs = "#lon#lat" + paramStrs
    machines = machineStrs.split("#")[1:]
    params= paramStrs.split("#")[1:]
    tifPaths = tifPathStrs.split("#%")[1:]
    calculate(machines,params,tifPaths,excelFilePathStrs)