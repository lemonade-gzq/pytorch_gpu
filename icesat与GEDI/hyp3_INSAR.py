from hyp3_sdk import HyP3
hyp3 = HyP3(username='1103376967@qq.com', password='Gzq!96926325454')

# Sentinel-1 SLC 数据的 granule ID（参考图 + 次图）
ref_granule = 'S1A_IW_SLC__1SDV_20240922T230520_20240922T230547_055784_06D0BF_67DC'
sec_granule = 'S1A_IW_SLC__1SDV_20240910T230520_20240910T230547_055609_06C9DF_F4EB'

# 提交 InSAR 任务（含相干图）
job = hyp3.submit_insar_job(
    granule1=ref_granule,
    granule2=sec_granule,
    name='coherence_only_job',
    include_wrapped_phase=False,  # ❌ 不需要包裹相位
    include_inc_map=False,  # ❌ 不需要入射角图
    include_look_vectors=False,  # ❌ 不需要视角矢量
    include_dem=False,  # ❌ 不需要DEM
    apply_water_mask=True,  # ✅ 屏蔽水体区域，便于后处理
    include_displacement_maps=False,  # ❌ 不需要LOS or Vertical 位移图
    phase_filter_parameter=0.6  # ✅ 默认滤波强度
)

# 3. 等待处理完成
job = hyp3.watch(job)

# 4. 下载 ZIP 结果
job.download_files()