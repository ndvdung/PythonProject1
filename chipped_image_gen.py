import pandas as pd
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt


def cwt_to_image(nd_data, image_name, sample_rate=5120, wave_name="morl"):
    """小波变换，生成时频图
    :param: nd_data, 数据片段，一维ndarray
    :param:image_name, 保存的图片名称
    :param:sample_rate, 采样频率，默认 5120 HZ
    :param:wave_name, 小波名称
    """
    samples = len(nd_data)  # 采样点数

    # 根据采样频率生成时间轴 t
    t = np.linspace(0, samples / sample_rate, samples, endpoint=False)

    # 设置小波函数
    total_scal = 128  # 尺度长度
    fc = pywt.central_frequency(wave_name)  # 小波中心频率
    c_param = 2 * fc * total_scal
    scales = c_param / np.arange(total_scal, 0, -1)

    # 进行连续小波变换
    coefficients, frequencies = pywt.cwt(nd_data, scales, wave_name, 1 / sample_rate)
    amp = abs(coefficients)  # 小波系数矩阵绝对值

    # 生成图像并保存
    plt.figure(figsize=(4, 3))
    plt.contourf(t, frequencies, amp, cmap="jet")
    plt.axis("off")
    plt.savefig(image_name, bbox_inches="tight", pad_inches=0)
    plt.close()


file_path = "gearbox_data/Chipped_20_0.csv"

with open(file_path, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip() == "Data":
        data_start = i + 1
        break

df = pd.read_csv(file_path, sep="\t", skiprows=data_start, header=None)
df = df.dropna(axis=1, how="all")

print(df.head())
print(df.tail())
print(df.shape)

channel4 = df.iloc[:, 3].to_numpy()

print(channel4.shape)
print(channel4[:10])

output_folder = "chipped_20_0_images"
os.makedirs(output_folder, exist_ok=True)

segment_length = 1024

for i in range(100):

    start = i * segment_length
    end = start + segment_length

    segment = channel4[start:end]

    cwt_to_image(segment, os.path.join(output_folder, f"chipped_20_0_{i+1:03d}.png"))

print("Done!")
