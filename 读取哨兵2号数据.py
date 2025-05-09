import numpy as np
import matplotlib.pyplot as plt


def box_counting(rh_vector, scales):
    N = []
    for scale in scales:
        count = 0
        for i in range(0, len(rh_vector), scale):
            if np.any(rh_vector[i:i + scale]):
                count += 1
        N.append(count)
    return N


def calculate_fractal_dimension(rh_vector):
    # 归一化处理
    rh_vector = rh_vector / np.max(rh_vector)

    # 定义不同的尺度
    scales = np.arange(1, len(rh_vector) // 2, 1)

    # 计算每个尺度下的盒子数量
    N = box_counting(rh_vector, scales)

    # 拟合log-log曲线，计算分形维度
    log_scales = np.log(scales)
    log_N = np.log(N)
    coeffs = np.polyfit(log_scales, log_N, 1)

    fractal_dimension = -coeffs[0]
    return fractal_dimension


# 示例RH指标向量
rh_vector = np.random.rand(101)  # 随机生成一个RH向量作为示例

# 计算分形维度
fractal_dimension = calculate_fractal_dimension(rh_vector)
print("分形维度:", fractal_dimension)

# 绘制log-log图
scales = np.arange(1, len(rh_vector) // 2, 1)
N = box_counting(rh_vector, scales)
plt.plot(np.log(scales), np.log(N), 'o', label='Data')
plt.plot(np.log(scales), np.polyval(np.polyfit(np.log(scales), np.log(N), 1), np.log(scales)), label='Fit')
plt.xlabel('log(Scale)')
plt.ylabel('log(N)')
plt.legend()
plt.show()