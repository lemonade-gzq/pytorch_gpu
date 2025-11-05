# -*- coding: utf-8 -*-

import os

import h5py
import numpy as np
from pandas import concat
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

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
            all_beam_dfs = []  # 用于存储当前条带的所有 Beam 结果
            self.df = pd.DataFrame()
            # 打开 ATL03 文件
            with h5py.File(self.atl03Path + '\\' + self.atl03sets[atl03_idx], 'r') as f03:
                # 判断使用左侧或右侧激光束
                plus_energy = np.array(f03['ancillary_data/atlas_engineering/transmit/tx_pulse_energy'][:])
                self.gtx = ['gt1l', 'gt2l', 'gt3l'] if plus_energy[0, 0] > plus_energy[1, 0] else ['gt1r', 'gt2r','gt3r']
                print(f"Processing strip: ATL03_" + self.atl03sets[atl03_idx].split('_')[1] + self.gtx[0])

                # 打开 ATL08 文件
                with h5py.File(self.atl08Path + '\\' + self.atl08sets[atl08_idx], 'r') as f08:
                    print(f"Processing strip: ATL08_"+ self.atl08sets[atl08_idx].split('_')[1]+self.gtx[0] )

                    for beam in self.gtx:
                        if f'{beam}/signal_photons' not in f08:
                            print(f"Skipping {beam}: Missing in ATL08 (likely due to subsetting).")
                            continue
                        # === ATL03 光子级别数据 ===
                        lat_ph = np.array(f03[f'{beam}/heights/lat_ph'][:])
                        lon_ph = np.array(f03[f'{beam}/heights/lon_ph'][:])
                        h_ph = np.array(f03[f'{beam}/heights/h_ph'][:])
                        signal_conf = np.array(f03[f'{beam}/heights/signal_conf_ph'][:])
                        ph_index_beg = np.array(f03[f'{beam}/geolocation/ph_index_beg'][:])#每个分段中第一个光子对应的编号
                        segment_id = np.array(f03[f'{beam}/geolocation/segment_id'][:])
                        first_segment_id = segment_id[0] #其实分段的id

                        # --- 新增读取段距离和光子相对距离 ---
                        # segment_dist_x: 每个 20 米段相对于轨道起点的距离
                        segment_dist_x = np.array(f03[f'{beam}/geolocation/segment_dist_x'][:])
                        # dist_ph_along: 光子在该段内的相对距离
                        dist_ph_along = np.array(f03[f'{beam}/heights/dist_ph_along'][:])

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

                        # ===== Step 3: 计算光子索引 =====
                        ph_idx = ph_index_beg[relative_idx] + pc_indx - 2  # 各个段在 ATL03 中从第几个光子开始 + 光子编号 NASA 官方建议的方式
                        # ===== Step 4: 检查索引是否越界 =====
                        valid_ph_mask = (ph_idx >= 0) & (ph_idx < len(lat_ph))
                        ph_idx = ph_idx[valid_ph_mask]
                        pc_indx = pc_indx[valid_ph_mask]
                        pc_flag = pc_flag[valid_seg_mask][valid_ph_mask]
                        pc_seg_id = pc_seg_id[valid_seg_mask][valid_ph_mask]

                        # ===== Step 5: 提取 ATL03 中的属性 =====
                        selected_lat = lat_ph[ph_idx]
                        selected_lon = lon_ph[ph_idx]
                        selected_z = h_ph[ph_idx]
                        selected_conf = signal_conf[ph_idx][:, 0]

                        # ===== Step 6: 计算绝对沿轨距离 (相对于裁剪子集的起点) =====
                        # 提取光子在其 20m 段内的相对距离
                        selected_dist_along = dist_ph_along[ph_idx]
                        # 提取光子所属 20m 段的沿轨起始距离
                        # 获取每个光子所属段在 ATL03 数组中的索引（relative_idx_final）
                        relative_idx_final = relative_idx[valid_ph_mask]
                        # 获取这些段的沿轨起始距离
                        selected_segment_start_dist = segment_dist_x[relative_idx_final]
                        # 公式：绝对距离 = 段起点沿轨距离 + 光子相对段内距离
                        selected_abs_dist = selected_segment_start_dist + selected_dist_along
                        # 为了让裁剪数据的起点为 0 m，我们减去子集中的最小距离
                        min_abs_dist = np.min(selected_abs_dist)
                        selected_relative_abs_dist = selected_abs_dist - min_abs_dist



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
                            'along_track_dist':selected_relative_abs_dist[valid_segment],  # 沿轨距离
                        })

                        self.df = pd.concat([self.df, df_now], ignore_index=True)

            self.df = self.df.query('cloud_flag == 0')
            self.df = self.df.query('landcover == 111 or landcover == 113 or landcover == 112 or landcover == 114 or '
                                    'landcover == 115 or landcover == 116 or landcover == 121 or landcover == 123 or '
                                    'landcover == 122 or landcover == 124 or landcover == 125 or landcover == 126')
            self.df = self.df.query('snowcover == 1')
            self.df = self.df.query('classification != 0')
            self.df = self.df.query('conf == 4')
            self.df = pd.DataFrame(self.df, columns=['lat', 'lon', 'z', 'classification','along_track_dist'])

            csvFile = r"D:\\遥感数据\\国家公园星载激光雷达数据提取\\大熊猫\\2025\\" + self.atl03sets[atl03_idx]  + '.csv'
            self.df.to_csv(csvFile, index=False)
            print("Results saved to:", csvFile)




class DataVisualizer:
    CLASSIFICATION_MAP = {
        # 分类代码: [名称, 颜色]
        2: {'name': 'Ground', 'color': 'goldenrod', 's': 2},  # 地面
        1: {'name': 'Canopy', 'color': 'lightgreen', 's': 1},  # 冠层
        3: {'name': 'Top of Canopy', 'color': 'darkgreen', 's': 2},  # 冠层顶部
        # 0 和 4 可以作为背景色或不显示
        0: {'name': 'Noise', 'color': 'lightgray', 's': 0.5},
        4: {'name': 'Other', 'color': 'darkgray', 's': 0.5},
    }

    def __init__(self, df_data):
        if 'lat' not in df_data.columns or 'z' not in df_data.columns or 'classification' not in df_data.columns:
            raise ValueError("Input DataFrame must contain 'lat', 'z', and 'classification' columns.")

        # 将分类字段转换为整数，以匹配 CLASSIFICATION_MAP 的键
        self.df = df_data.copy()
        self.df['classification'] = self.df['classification'].astype(int)

    def plot_track(self, use_latitude=True, title=None):
        """
        绘制沿轨剖面图。
        :param use_latitude: True 使用 Latitude 作为 X 轴；False 使用 along_track_dist。
        :param title: 图表标题。
        """

        # 过滤只保留我们需要的分类点
        valid_classes = list(self.CLASSIFICATION_MAP.keys())
        df_filtered = self.df[self.df['classification'].isin(valid_classes)].copy()

        if df_filtered.empty:
            print("Warning: No points found after filtering for Ground, Canopy, or Top of Canopy.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # 确定 X 轴变量
        x_col = 'lat' if use_latitude else 'along_track_dist'
        x_label = 'Latitude (°)' if use_latitude else 'Along Track Distance (m)'

        # 按分类类型绘制散点图
        for c, props in self.CLASSIFICATION_MAP.items():
            mask = df_filtered['classification'] == c

            # 由于 Ground 点可能数量较少，我们使用不同的尺寸和颜色来突出
            if mask.any():
                ax.scatter(df_filtered[mask][x_col],
                           df_filtered[mask]['z'],
                           color=props['color'],
                           label=props['name'],
                           s=props['s'],  # 使用预定义的尺寸
                           alpha=0.8,
                           edgecolors='none')  # 移除边缘，使点更清晰

        # 设置图表元素
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Elevation (m)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('ICESat-2 ATL03/ATL08 Classified Photons', fontsize=14)

        # 设置图例，只显示 Ground, Canopy, Top of Canopy
        ax.legend(loc='upper right', markerscale=4)

        # 调整轴的范围，去除图示中顶部和底部的大量未分类光子
        # 自动调整Y轴，但可以手动设置：ax.set_ylim([0, 400])
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_3d(self, title=None):
        """
        绘制交互式三维光子分布图：X=Lon, Y=Lat, Z=Elevation。
        光子颜色基于 ATL08 classification。
        """

        # 1. 数据准备：过滤并确保分类是字符串类型以便 Plotly 正确着色
        valid_classes = list(self.CLASSIFICATION_MAP.keys())
        df_filtered = self.df[self.df['classification'].isin(valid_classes)].copy()

        # 将数字分类转换为描述性字符串，用于图例和颜色映射
        df_filtered['Class_Name'] = df_filtered['classification'].map(
            {k: v['name'] for k, v in self.CLASSIFICATION_MAP.items()}
        )

        # 2. 定义颜色映射，使 Plotly 的颜色与我们定义的颜色一致
        color_discrete_map = {
            v['name']: v['color'] for v in self.CLASSIFICATION_MAP.values() if
            v['name'] in df_filtered['Class_Name'].unique()
        }

        # 3. 使用 Plotly Express 创建 3D 散点图
        fig = px.scatter_3d(
            df_filtered,
            x='lon',
            y='lat',
            z='z',
            color='Class_Name',  # 根据分类名称着色
            color_discrete_map=color_discrete_map,  # 使用我们定义的颜色
            opacity=0.8,
            title=title if title else 'ICESat-2 3D Photon Cloud (Lon, Lat, Elevation)',
            labels={'lon': 'Longitude (°)', 'lat': 'Latitude (°)', 'z': 'Elevation (m)'}
        )

        # 4. 优化图表布局
        fig.update_traces(marker=dict(size=1.5))  # 减小点的大小，改善密集数据的显示效果

        # 优化轴比例，确保 Z 轴（高程）看起来不会过度压缩
        fig.update_layout(
            scene=dict(
                xaxis_title='Longitude (°)',
                yaxis_title='Latitude (°)',
                zaxis_title='Elevation (m)',
                # 可以强制设置 Z 轴的缩放比例，以更好地查看植被结构
                # zaxis=dict(scaleanchor="x", scaleratio=1), # 保持比例可能使得高程变化难以分辨，通常需要夸大Z轴
            ),
            margin=dict(r=10, l=10, b=10, t=40)
        )

        # 5. 显示图表（Plotly 会在浏览器或 Jupyter 中打开交互式窗口）
        fig.show()
if __name__ == "__main__":
    # 单个条带处理测试（批处理见末尾）:
    atl03Path = r"I:\icesat2\ATL03_大熊猫2025\test"
    atl08Path = r"I:\icesat2\ATL08_大熊猫2025\test"
    # 创建 ATLDataLoader 对象
    loader = ATLDataLoader(atl03Path, atl08Path)
    if 'df' in loader.__dict__ and not loader.df.empty:
        df_to_visualize = loader.df.copy()

        # 创建 DataVisualizer 对象并传入 DataFrame
        print("\nStarting Visualization...")

        # 提取文件名作为标题
        # 获取最后一个处理的文件名
        last_idx = loader.dateIndexhdf[-1][0]
        file_name = loader.atl03sets[last_idx].replace('.h5', '.csv')

        visualizer = DataVisualizer(df_to_visualize)
        visualizer.plot_track(use_latitude=False, title=file_name)

        # visualizer.plot_3d(title=f"3D Photon Cloud: {file_name}")
    else:
        print("\nError: DataLoader did not produce any valid DataFrame for visualization.")


