# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:20:18 2024

@author: zjm
"""

import z_preprocess_function as pre
import numpy as np
import pandas as pd
import os
import sys

def preProcess():

    geotiff_list_1, _ = pre.read_geotiff_folder(folder_path)

    # 裁剪影像范围，使几张影像行列数统一,并生成掩膜
    geotiff_list_2_cropped, _ = pre.crop_geotiff_list(geotiff_list_1)
    
    mask = np.where(geotiff_list_2_cropped[1]['data'] < -2.402823e+38, 0, 1)
    
    # 处理缺失数据值
    geotiff_list_3_removednodata, data_list_3_removednodata = pre.remove_nodata_values(geotiff_list_2_cropped)
    
    # 计算相关性矩阵
    corr_matrix = pre.visualize_corr_matrix(data_list_3_removednodata, show_image=False)
    
    # # 提取第2和第4张图像的第一主成分并合并成一张图像
    # pca_indexes = [2, 4]
    # pca_image = pre.pca_images(data_list_3_removednodata, pca_indexes)
    
    # # 更新字典列表和数据列表(删除2,4，增加pca)
    data_list_4_preprocess1 = data_list_3_removednodata.copy()
    geotiff_list_4_preprocess1 = geotiff_list_3_removednodata.copy()
    
    # for index in sorted(pca_indexes, reverse=True):
    #     del data_list_4_preprocess1[index]
    #     del geotiff_list_4_preprocess1[index]
    
    # geotiff_data_pca = geotiff_list_4_preprocess1[0].copy()
    # geotiff_data_pca['data'] = pca_image
    # geotiff_data_pca['file_name'] = "地形起伏度_坡度_pca"
    
    # data_list_4_preprocess1.append(pca_image)
    # geotiff_list_4_preprocess1.append(geotiff_data_pca)
    
    # 对影像做归一化
    geotiff_list_5_preprocess2, _ = pre.normalize_geotiff_data(geotiff_list_4_preprocess1, mask)
    
    # 遍历geotiff_list中的每个元素，分别输出到对应的文件中
    columnName = []
    columnName.append("lon")
    columnName.append("lat")
    for geotiff_data in geotiff_list_5_preprocess2:
        # 构造输出文件名
        output_filename = os.path.join(output_folder_path, geotiff_data['file_name'] + '.tif')
        columnName.append(geotiff_data['file_name'])
        # 输出地理影像数据到文件
        pre.write_geotiff(geotiff_data, output_filename)
    columnName.append("Level")
    
    # 遍历所有地理影像数据并提取指定经纬度位置上的像素值，保存为excel
    df_orgin = pd.read_excel(sampleXY_path)
    lonlat_list = df_orgin.iloc[:, :2].values
    pixel_values_list = pre.extract_pixel_values(geotiff_list_5_preprocess2, lonlat_list)
    pixel_values_list = np.array(pixel_values_list)
    pixel_values_list2 = np.transpose(pixel_values_list)
    
    df = pd.DataFrame(pixel_values_list2)
    df.insert(0, "lon", df_orgin.iloc[:, 0].values)
    df.insert(1, "lat", df_orgin.iloc[:, 1].values)
    df.insert(df.shape[1], "Level", df_orgin.iloc[:, 2].values)
    try:
        columnName[columnName.index("landuse2000")] = "landuse200"
        columnName[columnName.index("Maximum_snow depth")] = "Maximum_sn"
        columnName[columnName.index("SCDchangerate")] = "SCDchanger"
        columnName[columnName.index("SCMchangerate5")] = "SCMchanger"
        columnName[columnName.index("SCSchangerate5")] = "SCSchanger"
        columnName[columnName.index("SDSameMonthMean01")] = "SDSameMont"
        columnName[columnName.index("SDSameMonthMean02")] = "SDSameMo_1"
        columnName[columnName.index("SDSameMonthMean03")] = "SDSameMo_2"
        columnName[columnName.index("SDSameMonthStd01")] = "SDSameMo_3"
        columnName[columnName.index("SDSameMonthStd02")] = "SDSameMo_4"
        columnName[columnName.index("SDSameMonthStd03")] = "SDSameMo_5"
        columnName[columnName.index("XJmonthcgrt1")] = "XJmonthcgr"
        columnName[columnName.index("XJmonthcgrt2")] = "XJmonthc_1"
        columnName[columnName.index("XJmonthcgrt3")] = "XJmonthc_2"
        columnName[columnName.index("yearchangerate")] = "yearchange"
    except:
        columnName = columnName
    
    df.columns = columnName
    df.to_excel(sample_extract_path, index=False,columns=columnName)
    
    # 绘制图像与经纬度点
    # pre.plot_lonlat_on_geotiff(geotiff_list_5_preprocess2, lonlat_list)
    
    # 删除多余变量
    # keep_variable = "geotiff_list_5_preprocess2"
    # variable_names = [var for var in dir() if not var.startswith("_") and var != keep_variable]
    # for var in variable_names:
    #     delattr(sys.modules[__name__], var)


if __name__  ==  "__main__" :
    a = []
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    folder_path = a[0]
    # 输出影像路径
    output_folder_path = a[1]
    # 经纬度坐标路径
    sampleXY_path = a[2]
    # 输出的用于训练的特征的路径
    sample_extract_path = a[3]
    
    # folder_path = r"E:\TeacherLiu\TIf数据_坐标转换3"
    # output_folder_path = r"E:\TeacherLiu\result"
    # sampleXY_path = r"E:\TeacherLiu\lat.xls"
    # sample_extract_path = r"E:\TeacherLiu\lat2.xls"
    preProcess()
    

         