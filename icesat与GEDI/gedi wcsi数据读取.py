import h5py
import numpy as np
import pandas as pd
import os

beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011', 'BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']


def process_wsci(beam_data):
    beam = beam_data
    col = [f'wsci_a{j}' for j in range(7)]
    wsci_values = beam_data[col]
    wsci = (wsci_values.sum(axis=1) - wsci_values.max(axis=1) - wsci_values.min(axis=1)) / 5
    beam = beam.drop(columns=col)
    wsci = pd.DataFrame(wsci, columns=['wsci'])
    data_df = pd.concat([wsci, beam], axis=1)
    return data_df


def quality_select(df):
    # 开始筛选数据 query函数选择满足条件的
    df = df.query('wsci_quality_flag_a1 == 1')
    df = df.query('wsci_quality_flag_a2 == 1')
    df = df.query('wsci_quality_flag_a3 == 1')
    df = df.query('wsci_quality_flag_a4 == 1')
    df = df.query('wsci_quality_flag_a5 == 1')
    df = df.query('wsci_quality_flag_a6 == 1')
    df = df.query('wsci_quality_flag_a0 == 1')

    df = df.query('algorithm_run_flag_a1 == 1')
    df = df.query('algorithm_run_flag_a2 == 1')
    df = df.query('algorithm_run_flag_a3== 1')
    df = df.query('algorithm_run_flag_a4 == 1')
    df = df.query('algorithm_run_flag_a5 == 1')
    df = df.query('algorithm_run_flag_a6 == 1')
    df = df.query('algorithm_run_flag_a0 == 1')

    df = df.query('lat >= 30.75774029 and lat <= 31.3308249')
    df = df.query('lon >= 102.848198189 and lon <= 103.4257809060744')

    df = df.query('degrade_flag == 0')

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
            # 提取 sensitivity_aN 和 quality_flag_an 字段
            count = len(beam_data["geolocation/lat_lowestmode_a1"])
            wsci, quality_flag,algorithm_run_flag  =np.zeros((count, 7)), np.zeros((count, 7)), np.zeros((count, 7))
            lat = np.array(beam_data["lat_lowestmode"])
            lon= np.array(beam_data["lon_lowestmode"])
            degrade_flag = np.array(beam_data["degrade_flag"])
            # elev_outlier_flag = np.array(beam_data["elev_outlier_flag"])
            l2_quality_flag = np.array(beam_data["l2_quality_flag"])

            for i in range(7):
                if i <= 5:
                    wsci[:, i] = np.array(beam_data["wsci_prediction/wsci_a" + str(i + 1)])
                    quality_flag[:, i] = np.array(beam_data["wsci_prediction/wsci_quality_flag_a" + str(i + 1)])
                    algorithm_run_flag[:, i] = np.array(beam_data["wsci_prediction/algorithm_run_flag_a" + str(i + 1)])
                else:
                    wsci[:, i] = np.array(beam_data["wsci_prediction/wsci_a" + '10'])
                    quality_flag[:, i] = np.array(beam_data["wsci_prediction/wsci_quality_flag_a" + '10'])
                    algorithm_run_flag[:, i] = np.array(beam_data["wsci_prediction/algorithm_run_flag_a" + '10'])


            # 提取得到的数组转dataframe
            lat = pd.DataFrame(lat,columns=['lat'])
            lon = pd.DataFrame(lon, columns=['lon'])
            degrade_flag = pd.DataFrame(degrade_flag, columns=['degrade_flag'])
            # elev_outlier_flag = pd.DataFrame(elev_outlier_flag, columns=['elev_outlier_flag'])
            l2_quality_flag = pd.DataFrame(l2_quality_flag, columns=['l2_quality_flag'])
            wsci = pd.DataFrame(wsci, columns=[f'wsci_a{i}' for i in range(7)])
            quality_flag = pd.DataFrame(quality_flag, columns=[f'wsci_quality_flag_a{i}' for i in range(7)])
            algorithm_run_flag = pd.DataFrame(algorithm_run_flag, columns=[f'algorithm_run_flag_a{i}' for i in range(7)])

            # 刷新DataFrame用来存放数据
            df = pd.DataFrame()
            beam_data_df = pd.concat([df, lat, lon, degrade_flag, l2_quality_flag, wsci, quality_flag, algorithm_run_flag], axis=1)
            #  质量筛选
            beam_data_df = quality_select(beam_data_df)
            # 提取每个分位数（0-100），并计算了6种算法去除最值后的平均值，结果为点数*101的dataframe
            beam_data_df = process_wsci(beam_data_df)
            beam_data_df = pd.DataFrame(beam_data_df, columns=['lat', 'lon', 'wsci'])
            file_df = pd.concat([file_df, beam_data_df])

        f.close()
        output_excel_path = filepath + '\\' + str(hdf5_file).split('_')[3] + '_' + 'new.csv'
        file_df.to_csv(output_excel_path, index=False)
        print(f'所有数据已保存到 {output_excel_path}')


if __name__ == "__main__":
    atl03Path = r"D:\遥感数据\GEDI\L4C"
    # 创建 ATLDataLoader 对象
    GEDIDataLoader(atl03Path, )
