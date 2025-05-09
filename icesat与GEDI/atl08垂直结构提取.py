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

def calculate_vsc(canopy_h_metrics):
    rh_vector = np.where(canopy_h_metrics< 0, 0, canopy_h_metrics)  #  替换为false的值.where((60 > rh_data_selected) & (rh_data_selected > 1), 1)
    len = rh_vector.shape[0]
    Sh = np.zeros((len, 3))
    for i in range(len):
        row = rh_vector[i,:]
        cv = np.std(row) / (np.mean(row) + 0.001)
        S = pd.Series(row).skew()
        # 归一化处理，确保所有值之和为1
        row = row / np.sum(row + 0.001)
        # 计算Shannon熵
        E = (-np.sum(row * np.log(row + 0.001))) / np.log(np.unique(row).shape[0])  # 加1e-10防止log(0)
        Sh[i, 0],Sh[i, 1],Sh[i, 2] = cv, S, E
        # 将Sh作为最后一列加入到rh_data_selected这个dataframe中
    # Sh = (Sh-np.min(Sh))/(np.max(Sh)-np.min(Sh))
    Sh=pd.DataFrame(Sh,columns=('cv','skew','evenness'))
    return Sh


class ATLDataLoader:
    def __init__(self, atl08Path, gtx):
        # 初始化
        self.atl08Path = atl08Path
        self.gtx = gtx
        self.load()

    def load(self):
        # 读取ATL08分段数据
        f = h5py.File(self.atl08Path, 'r')
        atl08_lat = np.array(f[self.gtx + '/land_segments/latitude'][:])
        atl08_lon = np.array(f[self.gtx + '/land_segments/longitude'][:])
        atl08_canopy_h_metrics = np.array(f[self.gtx + '/land_segments/canopy/canopy_h_metrics'][:])
        h_te_interp = np.array(f[self.gtx + '/land_segments/terrain/h_te_interp'][:])
        dem_h = np.array(f[self.gtx + '/land_segments/dem_h'][:])
        atl08_cloud = np.array(f[self.gtx + '/land_segments/cloud_flag_atm'])
        atl08_landcover = np.array(f[self.gtx + '/land_segments/segment_landcover'])
        atl08_snow = np.array(f[self.gtx + '/land_segments/segment_snowcover'])
        h_canopy_uncertainty = np.array(f[self.gtx + '/land_segments/canopy/h_canopy_uncertainty'])
        f.close()


        # 创建DataFrame存放数据
        self.df = pd.DataFrame()
        self.df['lon'] = atl08_lon  # longitude
        self.df['lat'] = atl08_lat  # latitude
        self.df = pd.concat([self.df, calculate_vsc(atl08_canopy_h_metrics)], axis=1)
        self.df['cloud'] = atl08_cloud
        self.df['landcover'] = atl08_landcover
        self.df['snow'] = atl08_snow
        self.df['delt_h'] = h_te_interp - dem_h
        self.df['canopy_uncertainty'] = h_canopy_uncertainty
        print("df11.shape:", self.df.shape)
        self.df = self.df.query('lat >= 30.75774029 and lat <= 31.3308249')
        self.df = self.df.query('lon >= 102.848198189 and lon <= 103.4257809060744')
        print("df11.shape:", self.df.shape)
        self.df = self.df.query('cloud <= 0.2')
        print("df11.shape:", self.df.shape)
        self.df = self.df.query('landcover == 111 or landcover == 113 or landcover == 112 or landcover == 114 or '
                                'landcover == 115 or landcover == 116 or landcover == 121 or landcover == 123 or '
                                'landcover == 122 or landcover == 124 or landcover == 125 or landcover == 126')
        print("df11.shape:", self.df.shape)
        self.df = self.df.query('snow == 1')
        print("df11.shape:", self.df.shape)
        delt_h_std = self.df['delt_h'].std()
        self.df = self.df.query('delt_h < @delt_h_std')
        print("df11.shape:", self.df.shape)
        uncertainty_threshold = np.mean(h_canopy_uncertainty) + np.std(h_canopy_uncertainty)
        self.df = self.df.query('canopy_uncertainty < @uncertainty_threshold')
        print("df11.shape:", self.df.shape)
        self.df = pd.DataFrame(self.df, columns=['lat', 'lon', 'cv', 'skew','evenness'])

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

    atl08Path = r"D:\遥感数据\icesat_ATL08\2023\ATL08_20230106185230_02631806_006_02.h5"
    # 循环处理六个激光波束
    for beam in ['gt1r', 'gt2r', 'gt3r']:
        print("Processing beam:", beam)
        # 创建 ATLDataLoader 对象
        loader = ATLDataLoader(atl08Path, beam)
        # 将结果保存到 CSV 文件
        csvFile = r"D:\\遥感数据\\icesat_atl08_vsc\\" + atl08Path.split('_')[2] + beam + '_vsc.csv'
        loader.df.to_csv(csvFile, index=False)
        print("Results saved to:", csvFile)
