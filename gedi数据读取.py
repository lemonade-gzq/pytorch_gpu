import h5py
import numpy as np
import pandas as pd
import os

beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011', 'BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']


def calculate_shannon_entropy(rh_data_selected):
    df =(rh_data_selected).where((60 > rh_data_selected) & (rh_data_selected > 1), 1) #  替换为false的值
    len = df.shape[0]
    Sh = np.zeros((len, 1))
    for i in range(len):
        rh_vector = df.iloc[i,5:96]
        # 计算Shannon熵
        E = -np.sum(rh_vector * np.log(rh_vector ))  # 加1e-10防止log(0)
        Sh[i, 0] = E
        # 将Sh作为最后一列加入到rh_data_selected这个dataframe中
    # Sh = (Sh-np.min(Sh))/(np.max(Sh)-np.min(Sh))
    return Sh


def process_rh_data(beam_data):
    rh_data = np.zeros((beam_data["geolocation/rh_a1"].shape[0], 101))
    for i in range(101):
        rh_values = np.array([beam_data[f"geolocation/rh_a{j + 1}"][:, i] for j in range(6)])
        rh_data[:, i] = (rh_values.sum(axis=0) - rh_values.max(axis=0) - rh_values.min(axis=0)) / 4 / 100
    return pd.DataFrame(rh_data, columns=[f'{i}' for i in range(101)])


def quality_select(df):
    # 开始筛选数据 query函数选择满足条件的
    df = df.query('geo_quality_flag_a1 == 1')
    df = df.query('geo_quality_flag_a2 == 1')
    df = df.query('geo_quality_flag_a3 == 1')
    df = df.query('geo_quality_flag_a4 == 1')
    df = df.query('geo_quality_flag_a5 == 1')
    df = df.query('geo_quality_flag_a6 == 1')
    df = df.query('abs(elev_lowestmode_a1 - Tan_dem) < 50')
    df = df.query('abs(elev_lowestmode_a2 - Tan_dem) < 50')
    df = df.query('abs(elev_lowestmode_a3 - Tan_dem) < 50')
    df = df.query('abs(elev_lowestmode_a4 - Tan_dem) < 50')
    df = df.query('abs(elev_lowestmode_a5 - Tan_dem) < 50')
    df = df.query('abs(elev_lowestmode_a6 - Tan_dem) < 50')
    df = df.query('degrade == 0')
    df = df.query('sensitivity_a1 > 0.9')
    df = df.query('sensitivity_a2 > 0.9')
    df = df.query('sensitivity_a3 > 0.9')
    df = df.query('sensitivity_a4 > 0.9')
    df = df.query('sensitivity_a5 > 0.9')
    df = df.query('sensitivity_a6 > 0.9')
    df = df.query('quality_flag == 1')
    df = df.query('rx_algrunflag_a1 == 1')
    df = df.query('rx_algrunflag_a2 == 1')
    df = df.query('rx_algrunflag_a3 == 1')
    df = df.query('rx_algrunflag_a4 == 1')
    df = df.query('rx_algrunflag_a5 == 1')
    df = df.query('rx_algrunflag_a6 == 1')
    return df


def GEDIDataLoader(filepath):
    for hdf5_file in os.listdir(filepath):
        try:
            f = h5py.File(os.path.join(filepath, hdf5_file), 'r')  # 打开h5文件
            print('正在处理文件：' + str(hdf5_file))
        except:
            print('读取错误！')
            continue
        file_df = pd.DataFrame()
        for beam in beams:
            # 访问冠层高度数据集
            beam_data = f[str(beam)]
            # 仅读取全能量激光器
            if beam_data.attrs['description'] == 'Full power beam':
                # 提取 sensitivity_aN 和 quality_flag_an 字段
                count = len(beam_data["geolocation/sensitivity_a1"])
                rx_algrunflag, geo_quality_flag, sensitivity, elev_lowestmode = \
                    np.zeros((count, 6)), np.zeros((count, 6)), np.zeros((count, 6)), np.zeros((count, 6))
                for i in range(6):
                    sensitivity[:, i] = np.array(beam_data["geolocation/sensitivity_a" + str(i + 1)])
                    geo_quality_flag[:, i] = np.array(beam_data["geolocation/quality_flag_a" + str(i + 1)])
                    elev_lowestmode[:, i] = np.array(beam_data["geolocation/elev_lowestmode_a" + str(i + 1)])
                    rx_algrunflag[:, i] = np.array(beam_data["rx_processing_a" + str(i + 1) + '/rx_algrunflag'])
                # 提取得到的数组转dataframe
                geo_quality_flag = pd.DataFrame(geo_quality_flag,
                                                columns=[f'geo_quality_flag_a{i}' for i in range(1, 7)])
                sensitivity = pd.DataFrame(sensitivity, columns=[f'sensitivity_a{i}' for i in range(1, 7)])
                elev_lowestmode = pd.DataFrame(elev_lowestmode, columns=[f'elev_lowestmode_a{i}' for i in range(1, 7)])
                rx_algrunflag = pd.DataFrame(rx_algrunflag, columns=[f'rx_algrunflag_a{i}' for i in range(1, 7)])
                # 提取其他字段
                crown_lat = np.array(beam_data["lat_lowestmode"])
                crown_lon = np.array(beam_data["lon_lowestmode"])
                Tan_dem = np.array(beam_data['digital_elevation_model'])
                quality_flag = np.array(beam_data["rx_assess/quality_flag"])
                degrade_flag = np.array(beam_data["degrade_flag"])

                # 提取每个分位数（0-100），并计算了6种算法去除最值后的平均值，结果为点数*101的dataframe
                rh_data_selected = process_rh_data(beam_data)

                # 刷新DataFrame用来存放数据
                df = pd.DataFrame()
                df['Tan_dem'] = Tan_dem
                df['quality_flag'] = quality_flag
                df['rh95'] = rh_data_selected['95']
                df['sh'] = calculate_shannon_entropy(rh_data_selected)
                df['lon'] = crown_lon  # longitude
                df['lat'] = crown_lat  # latitude
                df['degrade'] = degrade_flag
                beam_data_df = pd.concat([df, geo_quality_flag, sensitivity, elev_lowestmode, rx_algrunflag], axis=1)
                #  质量筛选
                beam_data_df = quality_select(beam_data_df)

                beam_data_df = pd.DataFrame(beam_data_df, columns=['lat', 'lon', 'rh95', 'sh'])
                file_df = pd.concat([file_df, beam_data_df])

        f.close()
        output_excel_path = filepath + '\\' + str(hdf5_file).split('_')[3] + '_' + '.csv'
        file_df.to_csv(output_excel_path, index=False)
        print(f'所有数据已保存到 {output_excel_path}')


if __name__ == "__main__":
    atl03Path = r"D:\遥感数据\GEDI\wolong"
    # 创建 ATLDataLoader 对象
    GEDIDataLoader(atl03Path, )
