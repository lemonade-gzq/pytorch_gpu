from osgeo import gdal
from osgeo import osr
import numpy as np
import pandas as pd

rds = gdal.Open("D:\\遥感数据\\哨兵2卧龙\\S2A_MSIL2A_20230708T034541_N0509_R104_T48RUV_20230708T083604.SAFE\\MTD_MSIL2A.xml")
print(rds)
