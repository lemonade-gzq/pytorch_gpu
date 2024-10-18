import h5py
import numpy as np
import pandas as pd
import os

beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011', 'BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']


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

                geo_quality_flag = pd.DataFrame(geo_quality_flag, columns=[f'geo_quality_flag_a{i}' for i in range(1, 7)])
                sensitivity = pd.DataFrame(sensitivity, columns=[f'sensitivity_a{i}' for i in range(1, 7)])
                elev_lowestmode = pd.DataFrame(elev_lowestmode, columns=[f'elev_lowestmode_a{i}' for i in range(1, 7)])
                rx_algrunflag = pd.DataFrame(rx_algrunflag, columns=[f'rx_algrunflag_a{i}' for i in range(1, 7)])
                crown_lat = np.array(beam_data["lat_lowestmode"])
                crown_lon = np.array(beam_data["lon_lowestmode"])
                crown_rh1 = np.array(beam_data["geolocation/rh_a1"][:, 95])
                crown_rh2 = np.array(beam_data["geolocation/rh_a2"][:, 95])
                crown_rh3 = np.array(beam_data["geolocation/rh_a3"][:, 95])
                crown_rh4 = np.array(beam_data["geolocation/rh_a4"][:, 95])
                crown_rh5 = np.array(beam_data["geolocation/rh_a5"][:, 95])
                crown_rh6 = np.array(beam_data["geolocation/rh_a6"][:, 95])
                Tan_dem = np.array(beam_data['digital_elevation_model'])
                quality_flag = np.array(beam_data["rx_assess/quality_flag"])
                degrade_flag = np.array(beam_data["degrade_flag"])

                # 刷新DataFrame用来存放数据
                df = pd.DataFrame()
                df['rh1'] = crown_rh1
                df['rh2'] = crown_rh2
                df['rh3'] = crown_rh3
                df['rh4'] = crown_rh4
                df['rh5'] = crown_rh5
                df['rh6'] = crown_rh6
                df['Tan_dem'] = Tan_dem
                df['quality_flag'] = quality_flag
                df['rh95'] = (df.sum(axis=1) - df.max(axis=1) - df.min(axis=1)) / 4 / 100
                # df = df.drop(['rh1', 'rh2', 'rh3', 'rh4', 'rh5', 'rh6'], axis=1)
                df['lon'] = crown_lon  # longitude
                df['lat'] = crown_lat  # latitude
                df['degrade'] = degrade_flag
                beam_data_df = pd.concat([df, geo_quality_flag, sensitivity, elev_lowestmode, rx_algrunflag],
                                    axis=1)
                # 开始筛选数据 query函数选择满足条件的
                beam_data_df = beam_data_df.query('geo_quality_flag_a1 == 1')
                beam_data_df = beam_data_df.query('geo_quality_flag_a2 == 1')
                beam_data_df = beam_data_df.query('geo_quality_flag_a3 == 1')
                beam_data_df = beam_data_df.query('geo_quality_flag_a4 == 1')
                beam_data_df = beam_data_df.query('geo_quality_flag_a5 == 1')
                beam_data_df = beam_data_df.query('geo_quality_flag_a6 == 1')
                beam_data_df = beam_data_df.query('abs(elev_lowestmode_a1 - Tan_dem) < 50')
                beam_data_df = beam_data_df.query('abs(elev_lowestmode_a2 - Tan_dem) < 50')
                beam_data_df = beam_data_df.query('abs(elev_lowestmode_a3 - Tan_dem) < 50')
                beam_data_df = beam_data_df.query('abs(elev_lowestmode_a4 - Tan_dem) < 50')
                beam_data_df = beam_data_df.query('abs(elev_lowestmode_a5 - Tan_dem) < 50')
                beam_data_df = beam_data_df.query('abs(elev_lowestmode_a6 - Tan_dem) < 50')
                beam_data_df = beam_data_df.query('degrade == 0')
                beam_data_df = beam_data_df.query('sensitivity_a1 > 0.9')
                beam_data_df = beam_data_df.query('sensitivity_a2 > 0.9')
                beam_data_df = beam_data_df.query('sensitivity_a3 > 0.9')
                beam_data_df = beam_data_df.query('sensitivity_a4 > 0.9')
                beam_data_df = beam_data_df.query('sensitivity_a5 > 0.9')
                beam_data_df = beam_data_df.query('sensitivity_a6 > 0.9')
                beam_data_df = beam_data_df.query('quality_flag == 1')
                beam_data_df = beam_data_df.query('rx_algrunflag_a1 == 1')
                beam_data_df = beam_data_df.query('rx_algrunflag_a2 == 1')
                beam_data_df = beam_data_df.query('rx_algrunflag_a3 == 1')
                beam_data_df = beam_data_df.query('rx_algrunflag_a4 == 1')
                beam_data_df = beam_data_df.query('rx_algrunflag_a5 == 1')
                beam_data_df = beam_data_df.query('rx_algrunflag_a6 == 1')

                beam_data_df = pd.DataFrame(beam_data_df, columns=['lat', 'lon', 'rh95'])
                file_df = pd.concat([file_df,beam_data_df])
        f.close()
        output_excel_path = filepath + '\\' + str(hdf5_file).split('_')[3] + '_' +  '.csv'
        file_df.to_csv(output_excel_path, index=False)
        print(f'所有数据已保存到 {output_excel_path}')



if __name__ == "__main__":
    atl03Path = r"D:\遥感数据\GEDI\wolong"
    # 创建 ATLDataLoader 对象
    GEDIDataLoader(atl03Path, )
