from osgeo import gdal
from osgeo import osr
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA


def read_geotiff(filename):
    """
    读取geotiff格式的地理影像文件
    :param filename: 地理影像文件名（包括路径和扩展名）
    :return: 包含地理影像数据及其各种信息的字典
    """
    # 获取文件名，不包括路径和扩展名
    file_name = os.path.splitext(os.path.basename(filename))[0]

    dataset = gdal.Open(filename)  # 打开文件
    band = dataset.GetRasterBand(1)  # 获取第一个波段（假设只有一个波段）
    geotransform = dataset.GetGeoTransform()  # 获取仿射变换信息
    projection = dataset.GetProjection()  # 获取投影信息
    data = band.ReadAsArray()  # 读取地理影像数据
    nodata = band.GetNoDataValue()  # 获取nodata值
    xsize = dataset.RasterXSize  # 获取水平方向像元数目
    ysize = dataset.RasterYSize  # 获取垂直方向像元数目
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(projection)
    geosrs = prosrs.CloneGeogCS()

    # 将数据及其各种信息存储到字典中
    geotiff_data = {
        'file_name': file_name,
        'data': data,
        'geotransform': geotransform,
        'projection': projection,
        'nodata': nodata,
        'xsize': xsize,
        'ysize': ysize,
        'prosrs': prosrs,
        'geosrs': geosrs
    }

    dataset = None  # 关闭数据集

    return geotiff_data


def read_geotiff_folder(folder_path):
    """
    读取一个文件夹中所有的geotiff格式地理影像文件
    :param folder_path: 文件夹路径
    :return: 包含所有地理影像数据及其各种信息的字典列表和数据列表
    """
    # 存储所有地理影像数据及其各种信息的字典列表
    geotiff_list = []

    # 存储所有地理影像数据的列表
    data_list = []

    # 获取文件夹中所有文件的路径
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]

    # 遍历所有文件并读取数据和信息
    for file_path in file_list:
        geotiff_data = read_geotiff(file_path)
        geotiff_list.append(geotiff_data)
        data_list.append(geotiff_data['data'])

    return geotiff_list, data_list


def write_geotiff(geotiff_data, output_filename):
    """
    将地理影像数据及其各种信息写入到geotiff格式的文件中
    :param geotiff_data: 包含地理影像数据及其各种信息的字典
    :param output_filename: 输出文件名（包括路径和扩展名）
    :return: None
    """
    driver = gdal.GetDriverByName('GTiff')  # 获取GTiff驱动
    rows, cols = geotiff_data['data'].shape  # 获取影像的行列数

    # 创建输出文件
    dataset = driver.Create(output_filename, cols, rows, 1, gdal.GDT_Float32)

    # 设置仿射变换信息和投影信息
    dataset.SetGeoTransform(geotiff_data['geotransform'])
    dataset.SetProjection(geotiff_data['projection'])

    # 获取第一个波段并将数据写入文件
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(geotiff_data['nodata'])
    band.WriteArray(geotiff_data['data'])

    dataset = None  # 关闭数据集


def crop_geotiff_list(geotiff_list):
    """
    裁剪所有地理影像数据，使它们的行列数相同
    :param geotiff_list: 包含所有地理影像数据及其各种信息的字典列表
    :return: 包含所有裁剪后的地理影像数据及其各种信息的字典列表和数据列表
    """
    # 找出最小行列数
    min_xsize = min([geotiff_data['xsize'] for geotiff_data in geotiff_list])
    min_ysize = min([geotiff_data['ysize'] for geotiff_data in geotiff_list])

    # 裁剪所有地理影像数据
    cropped_geotiff_list = []
    cropped_data_list = []
    for geotiff_data in geotiff_list:
        # 计算裁剪的范围
        x_offset = (geotiff_data['xsize'] - min_xsize) // 2
        y_offset = (geotiff_data['ysize'] - min_ysize) // 2
        x_end = x_offset + min_xsize
        y_end = y_offset + min_ysize

        # 裁剪数据
        cropped_data = geotiff_data['data'][y_offset:y_end, x_offset:x_end]

        # 更新影像信息
        geotiff_data['data'] = cropped_data
        geotiff_data['xsize'] = min_xsize
        geotiff_data['ysize'] = min_ysize

        # 存储裁剪后的影像数据及其信息
        cropped_geotiff_list.append(geotiff_data)

        cropped_data_list.append(cropped_data)

    return cropped_geotiff_list, cropped_data_list


def remove_nodata_values(geotiff_list):
    """
    将数据列表中的所有缺失数据值替换为中位数
    :param geotiff_list: 包含所有地理影像数据及其各种信息的字典列表
    :return: 处理缺失值后的字典列表和数据列表
    """
    removenodata_geotiff_list = []
    removenodata_data_list = []
    for geotiff_data in geotiff_list:
        # 将缺失数据值替换为中位数
        median_value = np.median(geotiff_data['data'][geotiff_data['data'] > -2e+38])
        geotiff_data['data'][geotiff_data['data'] < -2e+38] = median_value

        # 存储处理后的影像数据及其信息
        removenodata_data_list.append(geotiff_data['data'])
        removenodata_geotiff_list.append(geotiff_data)

    return removenodata_geotiff_list, removenodata_data_list


def visualize_corr_matrix(data_list, show_image=True):
    # 将 data_list 中的每个二维数组展平为一维数组
    flat_data_list = [d.flatten() for d in data_list]

    # 将展平后的一维数组按行堆叠成一个二维数组
    stacked_data = np.vstack(flat_data_list)

    # 计算相关性矩阵
    corr_matrix = np.corrcoef(stacked_data)

    if show_image:
        # 将矩阵展示为图像
        plt.imshow(corr_matrix)
        plt.colorbar()
        plt.show()

    return corr_matrix


def pca_images(images_list, indices):
    """
    对列表中的图像进行PCA，并返回主成分图像
    :param images_list: 包含多个图像的列表
    :param indices: 要组合并进行PCA的图像的索引列表
    :return: 保留第一主成分的图像
    """
    # 将要组合并进行PCA的图像组合成一个Numpy数组
    combined_image = np.dstack([images_list[i] for i in indices])

    # 将数组转换为二维
    flattened_image = combined_image.reshape(-1, combined_image.shape[-1])

    # 使用PCA提取主成分
    pca = PCA(n_components=1)
    transformed_data = pca.fit_transform(flattened_image)

    # 将变换后的数据转换回原始的形状
    transformed_image = transformed_data.reshape(combined_image.shape[:-1])

    return transformed_image


def normalize_geotiff_data(geotiff_list, mask=None):
    """
    对每个地理影像数据进行归一化处理
    :param geotiff_list: 包含所有地理影像数据及其各种信息的字典列表
    :param mask: 掩膜数组，默认为None
    :return: 归一化后的地理影像数据及其各种信息的字典列表和数据列表
    """
    # 存储归一化后的所有地理影像数据及其各种信息的字典列表
    normalized_geotiff_list = []

    # 存储归一化后的所有地理影像数据的列表
    normalized_data_list = []

    # 遍历所有地理影像数据并归一化处理
    for geotiff_data in geotiff_list:
        data = geotiff_data['data']
        # 之前不知为何需要 +0.5 现在予以删除
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        if mask is not None:
            normalized_data = normalized_data * mask
        normalized_geotiff_data = {
            'file_name': geotiff_data['file_name'],
            'data': normalized_data,
            'geotransform': geotiff_data['geotransform'],
            'projection': geotiff_data['projection'],
            'nodata': geotiff_data['nodata'],
            'xsize': geotiff_data['xsize'],
            'ysize': geotiff_data['ysize'],
            'prosrs': geotiff_data['prosrs'],
            'geosrs': geotiff_data['geosrs']
        }
        normalized_geotiff_list.append(normalized_geotiff_data)
        normalized_data_list.append(normalized_data)

    return normalized_geotiff_list, normalized_data_list


def lonlat2geo(geotiff_data, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param geotiff_data: 地理影像数据及其各种信息的字典
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标 (lat, lat)对应的投影坐标
    '''
    prosrs = geotiff_data['prosrs']
    geosrs = geotiff_data['geosrs']
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    # 这里输入别为纬度与经度。输出的也是纬度与经度对应的y，x。
    coords = ct.TransformPoint(lat, lon)
    return coords[:2]


def extract_pixel_values(geotiff_list, lonlat_list):
    """
    提取给定经纬度坐标对应的像素值
    :param geotiff_list: 包含地理影像数据及其各种信息的字典列表
    :param lonlat_list: 经纬度坐标列表，第一列为经度，第二列为纬度
    :return: 包含所有提取像素值的列表
    """
    # 存储所有提取像素值的列表
    pixel_values_list = []

    for geotiff_data in geotiff_list:
        # 存储当前影像的像素值列表
        current_pixel_values = []

        for lonlat in lonlat_list:
            # 将经纬度坐标转换为当前影像的投影坐标
            lon, lat = lonlat
            y, x = lonlat2geo(geotiff_data, lon, lat)
            # x, y = lon, lat
            # 将投影坐标转换为像素坐标
            gt = geotiff_data['geotransform']
            px = int((x - gt[0]) / gt[1])
            py = int((y - gt[3]) / gt[5])

            # 提取像素值
            pixel_value = geotiff_data['data'][py][px]

            # 如果像素值为nodata值，则将其替换为NaN
            if pixel_value == geotiff_data['nodata']:
                #由于现有影像预处理将nodata设置为0，导致具有实际意义的0变为nodata，这里设置回来
                pixel_value = 0

            # 将当前像素值添加到列表中
            current_pixel_values.append(pixel_value)

        # 将当前影像的像素值列表添加到所有影像的像素值列表中
        pixel_values_list.append(current_pixel_values)

    return pixel_values_list


def plot_lonlat_on_geotiff(geotiff_list, lonlat_points):
    """
    将给定的经纬度坐标点绘制在第一张地理影像上
    :param geotiff_list: 包含所有地理影像数据及其各种信息的字典列表
    :param lonlat_points: 经纬度坐标点，一个二维nd.array，第一列是经度，第二列是纬度
    """
    # 获取第一张地理影像
    geotiff_data = geotiff_list[0]
    # 获取地理影像的数据和仿射变换信息
    data = geotiff_data['data']
    geotransform = geotiff_data['geotransform']

    # 将经纬度坐标转换为投影坐标
    x, y = [], []
    for point in lonlat_points:
        lon, lat = point
        py, px = lonlat2geo(geotiff_data, lon, lat)
        x.append(px)
        y.append(py)

    # 绘制地理影像
    plt.imshow(data, cmap=cm.gray, extent=[geotransform[0], geotransform[0] + geotransform[1] * data.shape[1],
                                           geotransform[3] + geotransform[5] * data.shape[0], geotransform[3]])

    # 绘制经纬度坐标点
    plt.scatter(x, y, s=20, c='r', marker='o', edgecolor='none')

    # 添加标题和坐标轴标签
    plt.title('Lon-Lat Points on Geotiff Image')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')

    # 显示图像
    plt.show()