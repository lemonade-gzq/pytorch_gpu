# -*- coding: utf-8 -*-
"""
# -*- coding: utf-8 -*-
@Time: 2024/4/14 11:19
@Author: LXX
@File: My_0308关联.py
@IDE：PyCharm
@Motto：ABC(Always Be Coding)
"""
import os

import h5py
import numpy as np
from pandas import concat
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
    def __init__(self, atl03Path, atl08Path):
        # 初始化
        self.atl03Path = atl03Path
        self.atl08Path = atl08Path
        self.gtx = []
        self.df_now = pd.DataFrame()



        self.atl03_files = [f for f in os.listdir(self.atl03Path) if f.lower().endswith(('.h5'))]
        self.atl03_files.sort()
        self.atl03sets = []
        for filename in self.atl03_files:
            # filepath = os.path.join(self.atl03Path, filename)
            self.atl03sets.append(filename)

        self.atl08_files = [f for f in os.listdir(self.atl08Path) if f.lower().endswith(('.h5'))]
        self.atl08_files.sort()
        self.atl08sets = []
        for filename in self.atl08_files:
            # filepath = os.path.join(self.atl08Path, filename)
            self.atl08sets.append(filename)

        self.dateIndexhdf = []
        for index, atl03 in enumerate(self.atl03sets):
            atl03data = atl03.split('_')[1]
            for atl08 in self.atl08sets:
                if atl03data in atl08:
                    self.dateIndexhdf.append((index, self.atl08sets.index(atl08)))
                    break
        self.load()




    def load(self):
        for atl03_idx, atl08_idx in self.dateIndexhdf:
            self.df = pd.DataFrame()
            # 打开 ATL03 文件
            with h5py.File(self.atl03Path + '\\' + self.atl03sets[atl03_idx], 'r') as f03:
                # 判断使用左侧或右侧激光束
                plus_energy = np.array(f03['ancillary_data/atlas_engineering/transmit/tx_pulse_energy'][:])
                self.gtx = ['gt1l', 'gt2l', 'gt3l'] if plus_energy[0, 0] > plus_energy[1, 0] else ['gt1r', 'gt2r','gt3r']

                # 打开 ATL08 文件
                with h5py.File(self.atl08Path + '\\' + self.atl08sets[atl08_idx], 'r') as f08:
                    for beam in self.gtx:
                        # === ATL03 光子级别数据 ===
                        lat_ph = np.array(f03[f'{beam}/heights/lat_ph'][:])
                        lon_ph = np.array(f03[f'{beam}/heights/lon_ph'][:])
                        h_ph = np.array(f03[f'{beam}/heights/h_ph'][:])
                        signal_conf = np.array(f03[f'{beam}/heights/signal_conf_ph'][:])
                        ph_index_beg = np.array(f03[f'{beam}/geolocation/ph_index_beg'][:])#每个分段中第一个光子对应的编号
                        segment_id = np.array(f03[f'{beam}/geolocation/segment_id'][:])
                        first_segment_id = segment_id[0] #其实分段的id

                        # === ATL08 光子分类数据 ===
                        pc_indx = np.array(f08[f'{beam}/signal_photons/classed_pc_indx'][:]) # 光子在段中第几个
                        pc_flag = np.array(f08[f'{beam}/signal_photons/classed_pc_flag'][:])
                        pc_seg_id = np.array(f08[f'{beam}/signal_photons/ph_segment_id'][:]) #光子所在分段的索引

                        # === ATL08 分段属性数据（land_segments） ===
                        segment_id_beg = np.array(f08[f'{beam}/land_segments/segment_id_beg'][:])
                        segment_id_end = np.array(f08[f'{beam}/land_segments/segment_id_end'][:])
                        cloud_flag = np.array(f08[f'{beam}/land_segments/cloud_flag_atm'][:])
                        landcover = np.array(f08[f'{beam}/land_segments/segment_landcover'][:])
                        snowcover = np.array(f08[f'{beam}/land_segments/segment_snowcover'][:])

                        # ==== 光子过滤并与 ATL03 对齐 ====
                        relative_idx = pc_seg_id - first_segment_id #确定光子所在段在atl03的相对段的位置
                        # ===== Step 2: 确保段号在合法范围（ph_index_beg 的索引范围）=====
                        valid_seg_mask = (relative_idx >= 0) & (relative_idx < len(ph_index_beg))
                        relative_idx = relative_idx[valid_seg_mask]
                        pc_indx = pc_indx[valid_seg_mask]
                        pc_flag = pc_flag[valid_seg_mask]
                        pc_seg_id = pc_seg_id[valid_seg_mask]
                        # ===== Step 3: 计算光子索引 =====
                        ph_idx = ph_index_beg[relative_idx] + pc_indx - 2  # 各个段在 ATL03 中从第几个光子开始 + 光子编号 NASA 官方建议的方式
                        # ===== Step 4: 检查索引是否越界 =====
                        valid_ph_mask = (ph_idx >= 0) & (ph_idx < len(lat_ph))
                        ph_idx = ph_idx[valid_ph_mask]
                        pc_indx = pc_indx[valid_ph_mask]
                        pc_flag = pc_flag[valid_ph_mask]
                        pc_seg_id = pc_seg_id[valid_ph_mask]

                        # ===== Step 5: 提取 ATL03 中的属性 =====
                        selected_lat = lat_ph[ph_idx]
                        selected_lon = lon_ph[ph_idx]
                        selected_z = h_ph[ph_idx]
                        selected_conf = signal_conf[ph_idx][:, 0]

                        # === land_segments 信息映射到光子级别 ===
                        # 创建空数组存每个光子对应 land_segment 的索引
                        segment_indices = np.full_like(pc_seg_id, -1)
                        # 遍历所有 land_segments 区间，分配给落在该区间内的光子索引
                        for i, (beg, end) in enumerate(zip(segment_id_beg, segment_id_end)):
                            mask = (pc_seg_id >= beg) & (pc_seg_id <= end)
                            segment_indices[mask] = i  # 第 i 个 land segment
                        valid_segment = segment_indices >= 0

                        # 创建临时 DataFrame
                        df_now = pd.DataFrame({
                            'lat': selected_lat[valid_segment],
                            'lon': selected_lon[valid_segment],
                            'z': selected_z[valid_segment],
                            'conf': selected_conf[valid_segment],
                            'classification': pc_flag[valid_segment],
                            'cloud_flag': cloud_flag[segment_indices[valid_segment]],
                            'landcover': landcover[segment_indices[valid_segment]],
                            'snowcover': snowcover[segment_indices[valid_segment]],
                        })

                        self.df = pd.concat([self.df, df_now], ignore_index=True)

            self.df = self.df.query('cloud_flag == 0')
            self.df = self.df.query('landcover == 111 or landcover == 113 or landcover == 112 or landcover == 114 or '
                                    'landcover == 115 or landcover == 116 or landcover == 121 or landcover == 123 or '
                                    'landcover == 122 or landcover == 124 or landcover == 125 or landcover == 126')
            self.df = self.df.query('snowcover == 1')
            self.df = self.df.query('classification != 0')
            self.df = self.df.query('conf == 4')
            self.df = pd.DataFrame(self.df, columns=['lat', 'lon', 'z', 'classification'])

            csvFile = r"D:\\遥感数据\\国家公园星载激光雷达数据提取\\" + self.atl03sets[atl03_idx]  + '.csv'
            self.df.to_csv(csvFile, index=False)
            print("Results saved to:", csvFile)

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
    atl03Path = r"I:\icesat2\ATL03_东北虎豹2019"
    atl08Path = r"I:\icesat2\ATL08_东北虎豹2019"
    # 创建 ATLDataLoader 对象
    loader = ATLDataLoader(atl03Path, atl08Path)

