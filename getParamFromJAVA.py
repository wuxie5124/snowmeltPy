# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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
import sys
import os

def get_stacking(clf, x_train, y_train, x_test, n_folds= 5):
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


        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

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
        for low in range(len(param)-1):
            pn = param[low]
            tmp_list.append(data_train.iloc[row][pn])
        XList_train.append(tmp_list)
    ylist_train =data_train.Level.values
    for row in range(0, data_num2):
        tmp_list = []
        for low in range(len(param)-1):
            pn = param[low]
            tmp_list.append(data_test.iloc[row][pn])
        XList_test.append(tmp_list)
    ylist_test =data_test.Level.values   
    ylist_lon =data_test.lon.values
    ylist_lat =data_test.lat.values
    
    return np.array(XList_train),np.array(XList_test),ylist_train,ylist_test,ylist_lon,ylist_lat     

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
    
def calculate(machines, params, excelFilePathStrs):
    ML1 = getMLModule(machines[0])
    ML2 = getMLModule(machines[1])
    ML3 = getMLModule(machines[2])
    ML4 = getMLModule(machines[3])
    ML5 = getMLModule(machines[4])
    ML21 = getMLModule(machines[5])

    train_x, test_x, train_y, test_y,lon_y,lat_y = loadData(excelFilePathStrs ,params)
    train_sets = []
    test_sets = []
    for clf in [ML1, ML2, ML3, ML4, ML5]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
        train_sets.append(train_set)
        test_sets.append(test_set)
    leval1_array = np.array(test_sets).T
    leval1_pd= pd.DataFrame(leval1_array, columns=['RF','GBDT','SVM','KNN','XGB']) 
    directory = os.path.dirname(excelFilePathStrs)
    leval1_pd.to_csv(directory + '\\Firstclassflier.csv', index=None)
    print("第一层结果在：" + directory + '\\Firstclassflier.csv')
    meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)
    
    dt_model = ML21
    dt_model.fit(meta_train, train_y)
    df_predict = dt_model.predict(meta_test)
    
    value = np.array([lon_y,lat_y, test_y ,df_predict]).T
    
    leval2_pd= pd.DataFrame(value, columns=['LON','LAT','REAL','XGB']) 
    leval2_pd.to_csv(directory + '\\Secondclassflier.csv', index=None)
    print("第二层结果在：" + directory + '\\Secondclassflier.csv')
    value = np.array([lon_y,lat_y, test_y, df_predict]).T
    
    r2 = r2_score(test_y,df_predict)
    print("R方指数为：" + str(r2))
    
if __name__ == "__main__":
    a = []
    # excelFilePathStrs = r"C:\\Users\\zjm\\Desktop\\建模参数.xls"
    # machineStrs = r"#XGBoost%learning_rate&Default%n_estimators&Default%max_depth&Default%min_child_weight&Default%subsample&Default%colsample_bytree&Default%reg_alpha&Default%reg_lambda&Default#GBDT%n_estimators&Default%min_samples_split&Default%min_samples_leaf&Default%max_features&Default%subsample&Default%learning_rate&Default%random_state&Default#KNN%n_neighbors&Default%weights&uniform#RandomForest%n_estimators&Default%min_samples_split&Default%min_samples_leaf&Default%max_depth&Default%max_features&Default%random_state&Default#SVM%C&Default%kernel&rbf#XGBoost%learning_rate&Default%n_estimators&Default%max_depth&Default%min_child_weight&Default%subsample&Default%colsample_bytree&Default%reg_alpha&Default%reg_lambda&Default"
    # paramStrs = r"#lon#lat#Agricultur#Curve_numb#Distance_f#Elevation#GDP#landuse200#Maximum_sn#Particle_s#Population#Relative_E#Road_densi#Runoff_CV#SCDavg#SCDchanger#SCMchanger#SCSchanger#SDSameMont#SDSameMo_1#SDSameMo_2#SDSameMo_3#SDSameMo_4#SDSameMo_5#Slope#snowclass#Variance_c#Vegetation#XJAVHRR_av#XJmonthcgr#XJmonthc_1#XJmonthc_2#yearchange"    

    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    
    machineStrs = a[0]
    paramStrs = a[1]
    excelFilePathStrs = a[2]
    
    machines = machineStrs.split("#")[1:]
    params= paramStrs.split("#")[1:]
    calculate(machines,params,excelFilePathStrs)
    
    
    
    
        