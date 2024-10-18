# -*- coding: utf-8 -*-
"""
# -*- coding: utf-8 -*-
@Time: 2022/12/28 10:35
@Author: LXX
@File: ATL08提取数据.py
@IDE：PyCharm
@Motto：ABC(Always Be Coding)
"""

import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

data_path = 'D:\\遥感数据\\icesat_ATL08\\20230617\\'  # 待处理ATL08数据的文件夹路径
beams = ['gt1l', 'gt2l', 'gt3l', 'gt1r', 'gt2r', 'gt3r']  # 待提取的beam的名字

# 创建一个空的DataFrame来存储所有beam的数据
all_data = pd.DataFrame()

# 遍历所有HDF5数据
for hdf5_file in os.listdir(data_path):
    try:
        f = h5py.File(os.path.join(data_path, hdf5_file), 'r')  # 打开h5文件
        print('正在处理文件：' + str(hdf5_file))
    except:
        print('读取错误！')
        continue

    # 遍历所有需要提取的beam
    for beam in tqdm(beams, desc="处理进度"):
        # 这里因为有些ATL08数据的beam不全，需要跳过
        try:
            gt = f[str(beam)]
            land_segments = gt['land_segments']
        except:
            print(f'Beam {beam} 不存在于文件 {hdf5_file} 中')
            continue

        # 获取ATL08数据中需要的属性
        canopy = land_segments['canopy']
        terrain = land_segments['terrain']

        lat = land_segments['latitude']
        lon = land_segments['longitude']
        rgt = land_segments['rgt']  # 参考地面轨道
        cloud_flag_atm = land_segments['cloud_flag_atm']
        night = land_segments['night_flag']

        time_str = str(hdf5_file)[6:14]
        time_nor = [time_str for _ in lat]

        h_canopy_rel = canopy['h_canopy']
        h_canopy_abs = canopy['h_canopy_abs']  # 参考WGS84椭球体上方的高度
        h_canopy_uncertainty = canopy['h_canopy_uncertainty']
        h_surf = terrain['h_te_best_fit']

        # 创建新的数据结构来存储数据
        newdata_dic = {
            'latitude': lat,
            'longitude': lon,
            'rgt': rgt,
            'cloud_flag': cloud_flag_atm,
            'time': time_nor,
            'h_canopy_rel': h_canopy_rel,
            'h_canopy_abs': h_canopy_abs,
            'h_canopy_uncertainty': h_canopy_uncertainty,
            'h_surf': h_surf,
            'night': night,
            'beam': [beam for _ in lat]  # 添加beam列
        }

        newdata = pd.DataFrame(newdata_dic)

        # 剔除树高大于10000的噪声点
        newdata = newdata[newdata.h_canopy_rel < 10000]

        # 将当前beam的数据添加到总DataFrame中
        all_data = pd.concat([all_data, newdata], ignore_index=True)
        f.close()

    # 输出合并后的excel结果
    output_excel_path = 'D:\\遥感数据\\icesat_ATL08\\new\\' + hdf5_file.split('_')[1] +'_'+ hdf5_file.split('_')[-1].split('.')[0] + '.csv'
    all_data.to_csv(output_excel_path, index=False)
    print(f'所有数据已保存到 {output_excel_path}')
    all_data = pd.DataFrame()

                # #根据excel生成矢量结果
                # output_shp_name = os.path.join(output_shp_path, str(time_str) + '_' + str(beam) + '.shp')
                # driver = ogr.GetDriverByName('ESRI Shapefile')
                # data_source = driver.CreateDataSource(output_shp_name)
                #
                # proj = osr.SpatialReference()
                # proj.ImportFromEPSG(4326)
                # layer = data_source.CreateLayer(str(time_str) + '_' + str(beam),proj,ogr.wkbPoint)
                #
                #
                # #创建字段
                # field_name = ogr.FieldDefn("Latitude",ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("Longitude", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("RGT", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("Beam", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("Cloud_flag", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("Time", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("H_canopy_r", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("H_canopy_a", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("H_canopy_u", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                # field_name = ogr.FieldDefn("H_surf", ogr.OFTString)
                # field_name.SetWidth(20)
                # layer.CreateField(field_name)
                #
                #
                # for p in range(len(lat)):
                #     if p in record_list:
                #         continue
                #     # print(p)
                #     feature = ogr.Feature(layer.GetLayerDefn())
                #
                #     feature.SetField('Latitude',str(lat[p]))
                #     feature.SetField('Longitude', str(lon[p]))
                #     feature.SetField('RGT', str(rgt[p]))
                #     feature.SetField('Beam', str(beam_list[p]))
                #     feature.SetField('Cloud_flag', str(cloud_flag_atm[p]))
                #     feature.SetField('Time', str(time_nor[p]))
                #     feature.SetField('H_canopy_r', str(h_canopy_rel[p]))
                #     feature.SetField('H_canopy_a', str(h_canopy_abs[p]))
                #     feature.SetField('H_canopy_u', str(h_canopy_uncertainty[p]))
                #     feature.SetField('H_surf', str(h_surf[p]))
                #
                #
                #     point = ogr.Geometry(ogr.wkbPoint)
                #     point.AddPoint(float(lon[p]),float(lat[p]))
                #     feature.SetGeometry(point)
                #     layer.CreateFeature(feature)
                #     feature.Destroy()
                # data_source.Destroy()
