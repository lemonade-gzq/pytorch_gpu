# -*- coding: utf-8 -*-

import os
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

# 定义包含 CSV 文件的路径
# Define the directory containing the CSV files
FILE_DIRECTORY = r'D:\遥感数据\国家公园星载激光雷达数据提取\大熊猫\2023'

# 定义目标输出文件名
# Define the target output filename
OUTPUT_FILENAME = '2023_combined_lidar_data.csv'

# 定义 CSV 文件的预期列名
# Define the expected column names for the CSV files
EXPECTED_COLUMNS = ['lat', 'lon', 'z', 'classification']


def combine_csv_files(directory_path, output_name):
    """
    遍历指定路径下的所有 CSV 文件，合并它们的内容，并输出到新的 CSV 文件。

    Iterates through all CSV files in the specified directory, merges their
    contents, and outputs them to a new CSV file.
    """
    # 用于存储所有 CSV 数据帧的列表
    all_data = []

    # 统计成功读取的文件数量
    file_count = 0

    print(f"--- Starting CSV Combination ---")
    print(f"Source Directory: {directory_path}")

    # 遍历路径下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            full_path = os.path.join(directory_path, filename)

            try:
                # 尝试读取 CSV 文件
                # Try to read the CSV file
                df = pd.read_csv(full_path, header=0, usecols=EXPECTED_COLUMNS)

                # 检查数据框是否为空
                if df.empty:
                    print(f"Skipping empty file: {filename}")
                    continue

                # 将读取的数据添加到列表中
                all_data.append(df)
                file_count += 1
                print(f"Successfully loaded: {filename}")

            except EmptyDataError:
                print(f"⚠️ Warning: File is completely empty (no header/data): {filename}")
            except FileNotFoundError:
                print(f"❌ Error: File not found: {filename}")
            except ParserError as e:
                print(f"❌ Error: Failed to parse CSV file {filename}. Check for irregular formatting. Error: {e}")
            except Exception as e:
                print(f"❌ An unexpected error occurred while processing {filename}: {e}")

    # 检查是否读取到任何数据
    if not all_data:
        print("\n--- Process Finished ---")
        print("No valid CSV files were found or loaded. Output file was not created.")
        return

    # 整合所有数据帧
    print("\nCombining all dataframes...")
    combined_df = pd.concat(all_data, ignore_index=True)
    valid_landcover = [3, 4]

    # combined_df = combined_df[ (combined_df['classification'].isin(valid_landcover))] # canopy and top of canopy

    # 定义输出文件的完整路径
    output_path = os.path.join(directory_path, output_name)

    # 输出到新的 CSV 文件
    combined_df.to_csv(output_path, index=False)

    print("\n--- Process Completed Successfully ---")
    print(f"Total files combined: {file_count}")
    print(f"Total rows in output: {len(combined_df)}")
    print(f"Output saved to: {output_path}")


# 执行函数
if __name__ == "__main__":
    combine_csv_files(FILE_DIRECTORY, OUTPUT_FILENAME)
