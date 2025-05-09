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
import os

import collections

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
        type = np.array(f['/ancillary_data/atlas_engineering/transmit/tx_pulse_energy'][:])
        print(type[0:1000])
        atl03_lat = np.array(f[self.gtx + '/heights/lat_ph'][:])
        atl03_lon = np.array(f[self.gtx + '/heights/lon_ph'][:])
        atl03_ph_index_beg = np.array(f[self.gtx + '/geolocation/ph_index_beg'][:])
        atl03_segment_id = np.array(f[self.gtx + '/geolocation/segment_id'][:])
        atl03_heights = np.array(f[self.gtx + '/heights/h_ph'][:])  #光子的高程
        atl03_conf = np.array(f[self.gtx + '/heights/signal_conf_ph'][:])
        f.close()

        # 读取ATL08分段数据
        f = h5py.File(self.atl08Path, 'r')
        atl08_classed_pc_indx = np.array(f[self.gtx + '/signal_photons/classed_pc_indx'][:])
        atl08_classed_pc_flag = np.array(f[self.gtx + '/signal_photons/classed_pc_flag'][:])
        atl08_segment_id = np.array(f[self.gtx + '/signal_photons/ph_segment_id'][:])
        atl08_ph_h = np.array(f[self.gtx + '/signal_photons/ph_h'][:])  #内插地表上方的光子高度
        f.close()

        # 利用光子区间号匹配(ph_segment_id=segment_id)，找到两组数据的相同区间
        '''返回两个数组，第一个表示atl08_segment_id中的每一个id在不在atl03_segment_id中，第二个数组表示在第几位'''
        atl03SegsIn08TF, atl03SegsIn08Inds = self.ismember(atl08_segment_id, atl03_segment_id)
        # data_count = collections.Counter(atl03SegsIn08TF)
        # print(data_count)

        # 获取ATL08分类的索引和标识值
        atl08classed_inds = atl08_classed_pc_indx[atl03SegsIn08TF]
        atl08classed_vals = atl08_classed_pc_flag[atl03SegsIn08TF]
        atl08_hrel = atl08_ph_h[atl03SegsIn08TF]

        # 确定ATL03数据的新映射
        atl03_ph_beg_inds = atl03SegsIn08Inds
        atl03_ph_beg_val = atl03_ph_index_beg[atl03_ph_beg_inds]
        newMapping = atl08classed_inds + atl03_ph_beg_val - 2

        # 获取输出数组的最大大小
        sizeOutput = newMapping[-1]

        # 用零预填充所有光子类阵列
        allph_classed = (np.zeros(sizeOutput + 1)) - 1
        allph_hrel = np.full(sizeOutput + 1, np.nan)

        # 加入ATL08分类信息
        allph_classed[newMapping] = atl08classed_vals
        allph_hrel[newMapping] = atl08_hrel

        # 匹配ATL03大小
        allph_classed = allph_classed[:len(atl03_heights)]
        allph_hrel = allph_hrel[:len(atl03_heights)]

        # 创建DataFrame存放数据
        self.df = pd.DataFrame()
        self.df['lon'] = atl03_lon[:len(allph_hrel)]  # longitude[:len(allph_hrel)]
        self.df['lat'] = atl03_lat[:len(allph_hrel)]  # latitude
        self.df['z'] = atl03_heights[:len(allph_hrel)]  # elevation
        self.df['h'] = allph_hrel[:len(allph_hrel)]  # 相对于参考面的高度
        self.df['conf'] = atl03_conf[:len(allph_hrel), 0]   # confidence flag（光子置信度）
        self.df['classification'] = allph_classed[:len(allph_hrel)] # atl08 classification（分类标识）

    def ismember(self, a_vec, b_vec, method_type='normal'):
        """ MATLAB equivalent ismember function """
        # 该函数主要用于判断一个数组中的元素是否存在于另一个数组中，并返回匹配的索引

        if (method_type.lower() == 'rows'):

            # 将a_vec转换为字符串数组
            a_str = a_vec.astype('str')
            b_str = b_vec.astype('str')

            # #将字符串连接成一维字符串数组
            for i in range(0, np.shape(a_str)[1]):
                a_char = np.char.array(a_str[:, i])
                b_char = np.char.array(b_str[:, i])
                if (i == 0):
                    a_vec = a_char
                    b_vec = b_char
                else:
                    a_vec = a_vec + ',' + a_char
                    b_vec = b_vec + ',' + b_char

        matchingTF = np.isin(a_vec, b_vec)
        common = a_vec[matchingTF]
        common_unique, common_inv = np.unique(common, return_inverse=True)  # common = common_unique[common_inv]
        b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
        common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
        matchingInds = common_ind[common_inv]

        return matchingTF, matchingInds

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



    # 测试文件路径
    atl03Path = r"D:\遥感数据\icesat_ATL03\ATL03_20230218044752_09111802_006_02.h5"
    atl08Path = r"D:\遥感数据\icesat_ATL08\202212_202304\ATL08_20230218044752_09111802_006_01.h5"
    # 循环处理六个激光波束
    for beam in ['gt1l', 'gt2l', 'gt3l']:
        print("Processing beam:", beam)
        # 创建 ATLDataLoader 对象
        loader = ATLDataLoader(atl03Path, atl08Path, beam)
        # 将结果保存到 CSV 文件
        csvFile = atl03Path.replace(".h5", '_' + beam + ".csv").replace(".hdf", '_' + beam + ".csv")
        loader.df.to_csv(csvFile, index=False)
        print("Results saved to:", csvFile)

