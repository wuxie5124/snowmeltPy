# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import json
import sys
import os

def gridSearchXGBoost(MLarray,X_train, y_train):
    paramlevel = {}
    count = len(MLarray)
    name = MLarray[0]
    other_params = {"eval_metric":'mae'}
    for i in range(1,count):
        param = MLarray[i].split('&')[0]
        value = MLarray[i].split('&')[1]
        if(value == "Default"):
            continue;
        other_params[param] = value
    cv_paramsDefault = {"eval_metric":['mae']}
    print(cv_paramsDefault)
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(
        estimator=model, param_grid=cv_paramsDefault, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
    optimized_GBM.fit(np.array(X_train), np.array(
        y_train))
    evalute_result = optimized_GBM.cv_results_['mean_test_score']
    listparam = list(cv_paramsDefault.keys())
    for i in range(len(listparam)):
        a = listparam[i]
        other_params[a] = optimized_GBM.best_params_[a]
    # print('所选取值：{0}'.format(optimized_GBM.best_params_))
    print('所选参数模型得分:{0}'.format(optimized_GBM.best_score_))        
    print(other_params)        
    default={}
    default["param"] = other_params
    default["score"] = optimized_GBM.best_score_
    
    params = []
    cv_params1 = {'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.7,0.9,1]}
    cv_params2 = {'n_estimators': [20, 50, 100, 120, 150,180,200]}
    cv_params3 = {'max_depth': [None,1, 2, 3, 4, 5, 7,  9, 11]}
    cv_params4 = {'min_child_weight': [None,1, 2, 3, 4, 5, 6]}
    cv_params5 = {'reg_alpha': [None,0.1, 0.2, 0.3, 0.4]}
    cv_params6 = {'reg_lambda': [None,1, 2, 4, 5, 7, 9]}
    cv_params7 = {'subsample': [None,0.1, 0.2, 0.4, 0.5, 0.7, 0.8]}
    cv_params8 = {'colsample_bytree': [None,0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]}

    other_params = {'learning_rate': 0.1, 'n_estimators': 40, 'max_depth': 4, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.5, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.2, 'reg_lambda': 5,"eval_metric":'mae'}
    params.append(cv_params1)
    params.append(cv_params2)
    params.append(cv_params3)
    params.append(cv_params4)
    params.append(cv_params6)
    params.append(cv_params7)
    params.append(cv_params8)
    for num in range(len(params)):
        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(
            estimator=model, param_grid=params[num], scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
        optimized_GBM.fit(np.array(X_train), np.array(
            y_train))
        evalute_result = optimized_GBM.cv_results_['mean_test_score']
        listparam = list(params[num].keys())
        for i in range(len(listparam)):
            a = listparam[i]
            other_params[a] = optimized_GBM.best_params_[a]
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))    
    print(other_params)     
    best ={}
    best["param"] = other_params
    best["score"] = optimized_GBM.best_score_
    paramlevel["type"] = "xgb" 
    paramlevel["default"] = default
    paramlevel["best"] = best
    return paramlevel

def gridSearchGBDT(MLarray,X_train,y_train):
    paramlevel = {}
    count = len(MLarray)
    name = MLarray[0]
    other_params = {}
    for i in range(1,count):
        param = MLarray[i].split('&')[0]
        value = MLarray[i].split('&')[1]
        if(value == "Default"):
            continue;
        other_params[param] = value
    cv_paramsDefault ={}    
    if(MLarray[1].split('&')[1] == "Default"):
        cv_paramsDefault[MLarray[1].split('&')[0]] = [20]
    else:
        cv_paramsDefault[MLarray[1].split('&')[0]] = [MLarray[1].split('&')[1]]
    model = GradientBoostingRegressor(**other_params)
    optimized_GBM = GridSearchCV(
        estimator=model, param_grid=cv_paramsDefault, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
    optimized_GBM.fit(np.array(X_train), np.array(
        y_train))
    evalute_result = optimized_GBM.cv_results_['mean_test_score']
    listparam = list(cv_paramsDefault.keys())
    for i in range(len(listparam)):
        a = listparam[i]
        other_params[a] = optimized_GBM.best_params_[a]
    # print('所选取值：{0}'.format(optimized_GBM.best_params_))
    print('所选参数模型得分:{0}'.format(optimized_GBM.best_score_)) 
    print(other_params)            
    default={}
    default["param"] = other_params
    default["score"] = optimized_GBM.best_score_
    
    params = []
    cv_params1 = {'n_estimators': [20, 50, 100, 120, 150,180,200]}
    cv_params2 = {'min_samples_split': [2, 5, 10, 20, 50,100    ]}
    cv_params3 = {'max_depth': [1, 2, 3, 4, 5, 7,  9, 11]}
    cv_params4 = {'min_samples_leaf': [1, 5, 10, 20, 50, 100]}
    cv_params5 = {'max_features': ["sqrt", "log2", 1, 3, 5, 7, 9, 11]}
    cv_params6 = {'subsample': [0.1, 0.2, 0.4, 0.5, 0.7, 0.8]}
    cv_params7 = {'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.7,0.9,1]}
    cv_params8 = {'random_state': [1,2,5]}

    other_params = {'n_estimators': 80, 'max_depth': 3, 'min_samples_split': 100, 'min_samples_leaf': 20,
                    'max_features': 'sqrt', 'subsample': 0.7, 'learning_rate': 0.05, 'random_state': 10}
    params.append(cv_params1)
    params.append(cv_params2)
    params.append(cv_params3)
    params.append(cv_params4)
    params.append(cv_params6)
    params.append(cv_params7)
    params.append(cv_params8)
    for num in range(len(params)):
        model = GradientBoostingRegressor(**other_params)
        optimized_GBM = GridSearchCV(
            estimator=model, param_grid=params[num], scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
        optimized_GBM.fit(np.array(X_train), np.array(
            y_train))
        evalute_result = optimized_GBM.cv_results_['mean_test_score']
        listparam = list(params[num].keys())
        for i in range(len(listparam)):
            a = listparam[i]
            other_params[a] = optimized_GBM.best_params_[a]
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))    
    print(other_params)     
    best ={}
    best["param"] = other_params
    best["score"] = optimized_GBM.best_score_
    paramlevel["type"] = "gdbt" 
    paramlevel["default"] = default
    paramlevel["best"] = best
    return paramlevel
def gridSearchKNN(MLarray,X_train,y_train):
    paramlevel = {}
    count = len(MLarray)
    name = MLarray[0]
    other_params = {}
    for i in range(1,count):
        param = MLarray[i].split('&')[0]
        value = MLarray[i].split('&')[1]
        if(value == "Default"):
            continue;
        other_params[param] = value
    cv_paramsDefault ={}    
    if(MLarray[1].split('&')[1] == "Default"):
        cv_paramsDefault[MLarray[1].split('&')[0]] = [1]
    else:   
        cv_paramsDefault[MLarray[1].split('&')[0]] = [MLarray[1].split('&')[1]]
    print(cv_paramsDefault)
    model = KNeighborsRegressor(**other_params)
    optimized_GBM = GridSearchCV(
        estimator=model, param_grid=cv_paramsDefault, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
    optimized_GBM.fit(np.array(X_train), np.array(
        y_train))
    evalute_result = optimized_GBM.cv_results_['mean_test_score']
    listparam = list(cv_paramsDefault.keys())
    for i in range(len(listparam)):
        a = listparam[i]
        other_params[a] = optimized_GBM.best_params_[a]
    # print('所选取值：{0}'.format(optimized_GBM.best_params_))
    print('所选参数模型得分:{0}'.format(optimized_GBM.best_score_)) 
    print(other_params)            
    default={}
    default["param"] = other_params
    default["score"] = optimized_GBM.best_score_
    
    params = []
    cv_params1 = {'n_neighbors': [1, 3, 5, 7, 10]}
    cv_params2 = {'weights': ["uniform", "distance"]}
    other_params = {'n_neighbors':1,'weights':"uniform"}
    params.append(cv_params1)
    params.append(cv_params2)
    for num in range(len(params)):
        model = KNeighborsRegressor(**other_params)
        optimized_GBM = GridSearchCV(
            estimator=model, param_grid=params[num], scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
        optimized_GBM.fit(np.array(X_train), np.array(
            y_train))
        evalute_result = optimized_GBM.cv_results_['mean_test_score']
        listparam = list(params[num].keys())
        for i in range(len(listparam)):
            a = listparam[i]
            other_params[a] = optimized_GBM.best_params_[a]
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))    
    print(other_params)     
    best ={}
    best["param"] = other_params
    best["score"] = optimized_GBM.best_score_
    paramlevel["type"] = "knn" 
    paramlevel["default"] = default
    paramlevel["best"] = best
    return paramlevel
def gridSearchRF(MLarray,X_train,y_train):
    paramlevel = {}
    count = len(MLarray)
    name = MLarray[0]
    other_params = {}
    for i in range(1,count):
        param = MLarray[i].split('&')[0]
        value = MLarray[i].split('&')[1]
        if(value == "Default"):
            continue;
        other_params[param] = value
    cv_paramsDefault ={}    
    if(MLarray[1].split('&')[1] == "Default"):
        cv_paramsDefault[MLarray[1].split('&')[0]] = [20]
    else:   
        cv_paramsDefault[MLarray[1].split('&')[0]] = [MLarray[1].split('&')[1]]
    print(cv_paramsDefault)
    model = RandomForestRegressor(**other_params)
    optimized_GBM = GridSearchCV(
        estimator=model, param_grid=cv_paramsDefault, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
    optimized_GBM.fit(np.array(X_train), np.array(
        y_train))
    evalute_result = optimized_GBM.cv_results_['mean_test_score']
    listparam = list(cv_paramsDefault.keys())
    for i in range(len(listparam)):
        a = listparam[i]
        other_params[a] = optimized_GBM.best_params_[a]
    # print('所选取值：{0}'.format(optimized_GBM.best_params_))
    print('所选参数模型得分:{0}'.format(optimized_GBM.best_score_))        
    print(other_params)     
    default={}
    default["param"] = other_params
    default["score"] = optimized_GBM.best_score_
    
    params = []

    cv_params1 = {'n_estimators': [20, 50, 100, 120, 150,180,200]}
    cv_params2 = {'min_samples_split': [2, 5, 10, 20, 50,100    ]}
    cv_params3 = {'min_samples_leaf': [1, 5, 10, 20, 50, 100]}    
    cv_params4 = {'max_depth': [None,1, 2, 3, 4, 5, 7,  9, 11]}
    cv_params5 = {'max_features': ["sqrt", "log2", 1, 3, 5, 7, 9, 11]}   
    cv_params6 = {'random_state': [1,2,5]}
    
    other_params = {'n_estimators': 10, 'max_depth': 1, 'min_samples_split': 100,
                    'min_samples_leaf': 10, 'max_features': 'sqrt', 'random_state': 1}
    params.append(cv_params1)
    params.append(cv_params2)
    params.append(cv_params3)
    params.append(cv_params4)
    params.append(cv_params5)
    params.append(cv_params6)
    for num in range(len(params)):
        model = RandomForestRegressor(**other_params)
        optimized_GBM = GridSearchCV(
            estimator=model, param_grid=params[num], scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
        optimized_GBM.fit(np.array(X_train), np.array(
            y_train))
        evalute_result = optimized_GBM.cv_results_['mean_test_score']
        listparam = list(params[num].keys())
        for i in range(len(listparam)):
            a = listparam[i]
            other_params[a] = optimized_GBM.best_params_[a]
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))    
    print(other_params)     
    best ={}
    best["param"] = other_params
    best["score"] = optimized_GBM.best_score_
    paramlevel["type"] = "rf" 
    paramlevel["default"] = default
    paramlevel["best"] = best
    return paramlevel
def gridSearchSVM(MLarray,X_train,y_train):
    paramlevel = {}
    count = len(MLarray)
    name = MLarray[0]
    other_params = {}
    for i in range(1,count):
        param = MLarray[i].split('&')[0]
        value = MLarray[i].split('&')[1]
        if(value == "Default"):
            continue;
        other_params[param] = value
    cv_paramsDefault ={}    
    if(MLarray[1].split('&')[1] == "Default"):
        cv_paramsDefault[MLarray[1].split('&')[0]] = [0.1]
    else:   
        cv_paramsDefault[MLarray[1].split('&')[0]] = [MLarray[1].split('&')[1]]
    print(cv_paramsDefault)
    model = SVR(**other_params)
    optimized_GBM = GridSearchCV(
        estimator=model, param_grid=cv_paramsDefault, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
    optimized_GBM.fit(np.array(X_train), np.array(
        y_train))
    evalute_result = optimized_GBM.cv_results_['mean_test_score']
    listparam = list(cv_paramsDefault.keys())
    for i in range(len(listparam)):
        a = listparam[i]
        other_params[a] = optimized_GBM.best_params_[a]
    print('所选取值：{0}'.format(optimized_GBM.best_params_))
    print('所选参数模型得分:{0}'.format(optimized_GBM.best_score_))        
    default={}
    default["param"] = other_params
    default["score"] = optimized_GBM.best_score_
    
    params = []
    cv_params1 = {'C': [0.1, 0.5, 1, 2, 5,10]}
    cv_params2 = {'kernel': ["rbf", "linear", "poly", "sigmoid"]}
    other_params = {'C':0.1,'kernel':"linear" }
    params.append(cv_params1)
    params.append(cv_params2)
    for num in range(len(params)):
        model =SVR(**other_params)
        optimized_GBM = GridSearchCV(
            estimator=model, param_grid=params[num], scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
        optimized_GBM.fit(np.array(X_train), np.array(
            y_train))
        evalute_result = optimized_GBM.cv_results_['mean_test_score']
        listparam = list(params[num].keys())
        for i in range(len(listparam)):
            a = listparam[i]
            other_params[a] = optimized_GBM.best_params_[a]
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_)) 
    print(other_params)     
    best ={}
    best["param"] = other_params
    best["score"] = optimized_GBM.best_score_
    paramlevel["type"] = "svm" 
    paramlevel["default"] = default
    paramlevel["best"] = best    
    return paramlevel

def writeJson(params):
    with open(jsonPath,"w") as f:
        json.dump(params,f)
    
def gridSearch(machines,params,excelFilePathStrs):
    train_x, test_x, train_y, test_y,lon_y,lat_y = loadData(excelFilePathStrs ,params)
    train_y = le.fit_transform(train_y)
    yyy = le.inverse_transform(train_y)
    # gridSearchXGBoost(machines[0],train_x,train_y)
    json_dict = {}
    for ilevel in range(len(machines)):
        json_dict[ilevel] = getMLModule(machines[ilevel],train_x,train_y) 
    print(json_dict)
    writeJson(json_dict)
    
def getMLModule(machinestr,train_x,train_y):
    MLarray = machinestr.split("%")
    name = MLarray[0]
    paramlevel = {}
    if(name == 'XGBoost'):
        paramlevel = gridSearchXGBoost(MLarray,train_x,train_y)
    elif(name == 'GBDT'):
        paramlevel = gridSearchGBDT(MLarray,train_x,train_y)
    elif(name == 'KNN'):
        paramlevel = gridSearchKNN(MLarray,train_x,train_y)
    elif(name == 'RandomForest'):
        paramlevel = gridSearchRF(MLarray,train_x,train_y)
    elif(name == 'SVM'):
        paramlevel =gridSearchSVM(MLarray,train_x,train_y)
    return paramlevel
    
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

if __name__ == "__main__":
    a = []
    # excelFilePathStrs = r"C:\\Users\\zhangjunmin\\Desktop\\11\\建模参数1.xls"
    # machineStrs = r"#XGBoost%learning_rate&Default%n_estimators&Default%max_depth&Default%min_child_weight&Default%subsample&Default%colsample_bytree&Default%reg_alpha&Default%reg_lambda&Default#GBDT%n_estimators&Default%min_samples_split&Default%min_samples_leaf&Default%max_features&Default%subsample&Default%learning_rate&Default%random_state&Default#KNN%n_neighbors&Default%weights&uniform#RandomForest%n_estimators&Default%min_samples_split&Default%min_samples_leaf&Default%max_depth&Default%max_features&Default%random_state&Default#SVM%C&Default%kernel&rbf#XGBoost%learning_rate&Default%n_estimators&Default%max_depth&Default%min_child_weight&Default%subsample&Default%colsample_bytree&Default%reg_alpha&Default%reg_lambda&Default"
    # paramStrs = r"#lon#lat#Agricultur#Curve_numb#Distance_f#Elevation#GDP#landuse200#Maximum_sn#Particle_s#Population#Relative_E#Road_densi#Runoff_CV#SCDavg#SCDchanger#SCMchanger#SCSchanger#SDSameMont#SDSameMo_1#SDSameMo_2#SDSameMo_3#SDSameMo_4#SDSameMo_5#Slope#snowclass#Variance_c#Vegetation#XJAVHRR_av#XJmonthcgr#XJmonthc_1#XJmonthc_2#yearchange"    
    # jsonPath = r"C:\\Users\\zhangjunmin\\Desktop\\snow\\paramFile\\gridSearchParam.json"
    
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    
    machineStrs = a[0]
    paramStrs = a[1]
    excelFilePathStrs = a[2]
    jsonPath =a[3]
    
    machines = machineStrs.split("#")[1:]
    params= paramStrs.split("#")[1:]
    directory = os.path.dirname(excelFilePathStrs)
    gridSearch(machines,params,excelFilePathStrs)