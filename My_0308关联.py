# -*- coding: utf-8 -*-
"""
# -*- coding: utf-8 -*-
@Time: 2024/4/14 11:19
@Author: LXX
@File: My_0308关联.py
@IDE：PyCharm
@Motto：ABC(Always Be Coding)
"""

import h5py
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 使用matplotlib的颜色映射
cmap = plt.get_cmap('jet')

'''
关联 ATL08 与 ATL03 数据产品:

1.遍历 ATL08 、ATL03数据产品的每个光子；
2.利用光子区间号匹配(ph_segment_id=segment_id)，找到两组数据的相同区间；
3.逐个匹配每个光子，ATL03 数据的光子序号等于 ATL03 数据的起始光子序号与 ATL08 数据的相对光子序号相加
 (classed_pc_indx+ph_index_beg-1)

由此可以将ATL08的每个光子对应在ATL03 数据产品中
'''


class ATLDataLoader:
    def __init__(self, atl03Path, atl08Path, gtx):
        # 初始化
        self.atl03Path = atl03Path
        self.atl08Path = atl08Path
        self.gtx = gtx
        self.load()

    def load(self):
        '''
        步骤:
        1)打开并读取 ATL03 文件，提取经度、纬度、分段起始索引、分段ID、高度和信号置信度等数据;
        2)打开并读取 ATL08 文件，提取光子分类索引、分类标识、分段ID和高度等数据;
        3)使用 ismember 方法匹配 ATL03 和 ATL08 的分段，以确定相同分段之间的对应关系;
        4)根据匹配结果，确定新的映射关系，并根据映射关系将 ATL08 的分类信息和高度信息与 ATL03 对应起来;
        5)创建一个 DataFrame 存储处理后的数据。
        '''

        # 读取ATL03分段数据
        f = h5py.File(self.atl03Path, 'r')
        atl03_lat = np.array(f[self.gtx + '/heights/lat_ph'][:])
        atl03_lon = np.array(f[self.gtx + '/heights/lon_ph'][:])
        atl03_ph_index_beg = np.array(f[self.gtx + '/geolocation/ph_index_beg'][:])
        atl03_segment_id = np.array(f[self.gtx + '/geolocation/segment_id'][:])
        first_segment_id = atl03_segment_id[0]
        atl03_heights = np.array(f[self.gtx + '/heights/h_ph'][:])
        atl03_conf = np.array(f[self.gtx + '/heights/signal_conf_ph'][:])
        f.close()

        # 读取ATL08分段数据
        f = h5py.File(self.atl08Path, 'r')
        atl08_classed_pc_indx = np.array(f[self.gtx + '/signal_photons/classed_pc_indx'][:])
        atl08_classed_pc_flag = np.array(f[self.gtx + '/signal_photons/classed_pc_flag'][:])
        atl08_segment_id = np.array(f[self.gtx + '/signal_photons/ph_segment_id'][:])
        # atl08_ph_h = np.array(f[self.gtx + '/land_segments/canopy//h_canopy'][:])
        # atl08_cloud = np.array(f[self.gtx + '/land_segments/cloud_flag_atm'])
        # atl08_landcover = np.array(f[self.gtx + '/land_segments/segment_landcover'])
        # atl08_snow = np.array(f[self.gtx + '/land_segments/segment_snowcover'])
        f.close()

        atl08_20all_index = atl08_segment_id - first_segment_id
        # '''范围裁剪过则需要注意匹配长度'''
        # mask = ((atl08_20all_index > 0) & (atl08_20all_index <= len(atl03_ph_index_beg) - 1))
        # atl08_20all_index = atl08_20all_index[mask]
        # atl08_classed_pc_flag = atl08_classed_pc_flag[mask]
        # # atl08_ph_h = atl08_ph_h[mask]
        # atl08_classed_pc_indx = atl08_classed_pc_indx[mask]

        first_ph_index_all = atl03_ph_index_beg[atl08_20all_index] - 1
        ph_index_all = first_ph_index_all + atl08_classed_pc_indx - 1
        atl08_lat = atl03_lat[ph_index_all]
        atl08_lon = atl03_lon[ph_index_all]
        atl08_conf = atl03_conf[ph_index_all]
        atl03_heights = atl03_heights[ph_index_all]

        # 创建DataFrame存放数据
        self.df = pd.DataFrame()
        self.df['lon'] = atl08_lon  # longitude
        self.df['lat'] = atl08_lat  # latitude
        self.df['z'] = atl03_heights  # elevation
        # self.df['h'] = atl08_ph_h  # 相对于参考面的高度
        self.df['conf'] = atl08_conf[:, 0]  # confidence flag（光子置信度）
        self.df['classification'] = atl08_classed_pc_flag  # atl08 classification（分类标识）
        # self.df['cloud'] = atl08_cloud
        # self.df['landcover'] = atl08_landcover
        # self.df['snow'] = atl08_snow
        self.df = self.df.query('lat >= 30.75774029 and lat <= 31.3308249')
        self.df = self.df.query('lon >= 102.848198189 and lon <= 103.4257809060744')
        # self.df = self.df.query('cloud == 0')
        # self.df = self.df.query('landcover == 111 or landcover == 113 or landcover == 112 or landcover == 114 or '
        #                         'landcover == 115 or landcover == 116 or landcover == 121 or landcover == 123 or '
        #                         'landcover == 122 or landcover == 124 or landcover == 125 or landcover == 126')
        # self.df = self.df.query('snow == 1')
        self.df = self.df.query('classification != 0')
        self.df = self.df.query('conf == 3 or conf == 4')
        self.df = pd.DataFrame(self.df, columns=['lat', 'lon', 'z', 'classification'])

    def save_to_csv_with_progress(self, filename):
        # 没什么用
        total_rows = len(self.df)
        with tqdm(total=total_rows, desc="关联数据导出至CSV", unit="row") as pbar:
            start_time = time.time()
            self.df.to_csv(filename, index=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            pbar.set_postfix({"Time (s)": elapsed_time})
            pbar.update(total_rows)


class DataVisualizer:
    def __init__(self, csv_path, class_dict):
        self.df = pd.read_csv(csv_path)
        self.class_dict = class_dict

    def plot(self):
        fig, ax = plt.subplots()
        for c in self.class_dict.keys():
            mask = self.df['classification'] == c
            ax.scatter(self.df[mask]['lat'], self.df[mask]['z'],
                       color=self.class_dict[c]['color'], label=self.class_dict[c]['name'], s=1)
        ax.set_xlabel('Latitude (°)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Ground Track')
        ax.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    # 单个条带处理测试（批处理见末尾）:
    # atl03Path = "D:/other/icesat/text/1/shan/processed_ATL03_20200915100047_12530806_006_02.h5"
    # atl08Path = "D:/other/icesat/text/2/processed_ATL08_20200915100047_12530806_006_02.h5"
    atl03Path = r"D:\遥感数据\icesat_ATL03\2023\ATL03_20230520002727_09111902_006_02.h5"
    atl08Path = r"D:\遥感数据\icesat_ATL08\2023\ATL08_20230520002727_09111902_006_02.h5"
    # 循环处理六个激光波束
    for beam in ['gt1l', 'gt2l', 'gt3l']:
        print("Processing beam:", beam)
        # 创建 ATLDataLoader 对象
        loader = ATLDataLoader(atl03Path, atl08Path, beam)
        # 将结果保存到 CSV 文件
        csvFile = r"D:\遥感数据\icesat_ATL03_08\icesat" + atl03Path.split('_')[2] + beam + '.csv'
        loader.df.to_csv(csvFile, index=False)
        print("Results saved to:", csvFile)
