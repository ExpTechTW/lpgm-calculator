from obspy import read
from LPGMCalculator import LPGMCalculator
import matplotlib.pyplot as plt
import time
import numpy as np
from dvg_ringbuffer import RingBuffer
import matplotlib
matplotlib.use('TkAgg')  # 使用較快的後端

# 濾波器開關（注意：LPGMCalculator 內部已有高通濾波器，這裡的濾波僅用於顯示波形）
FILTER = True

# 播放偏移（秒）- 跳過前面的資料從指定時間開始播放
OFFSET = 60 * 6


def read_from_mseed(file_path):
    """從 mseed 文件讀取三個分量的數據，返回 (data_x, data_y, data_z, sample_rate)"""
    stream = read(file_path)
    sample_rate = int(stream[0].stats.sampling_rate)

    if len(stream) == 3:
        return stream[0].data, stream[1].data, stream[2].data, sample_rate

    # 根據通道名稱識別
    channels = {}
    for trace in stream:
        ch = trace.stats.channel
        if 'Z' in ch or ch.endswith('Z'):
            channels['Z'] = trace.data
        elif 'N' in ch or ch.endswith('N') or '1' in ch:
            channels['N'] = trace.data
        elif 'E' in ch or ch.endswith('E') or '2' in ch:
            channels['E'] = trace.data

    # 使用識別到的通道，否則使用前三個
    return (channels.get('N', stream[0].data),
            channels.get('E', stream[1].data if len(
                stream) > 1 else stream[0].data),
            channels.get('Z', stream[2].data if len(
                stream) > 2 else stream[0].data),
            sample_rate)


class BiquadFilter:
    """二階節濾波器類"""

    def __init__(self, num_coeffs, den_coeffs):
        if len(num_coeffs) != len(den_coeffs):
            raise ValueError(
                "num_coeffs and den_coeffs must have the same length")

        self.stages = []
        for num, den in zip(num_coeffs, den_coeffs):
            b0, b1, b2 = num
            a0, a1, a2 = den

            if a0 != 1.0:
                b0, b1, b2 = b0/a0, b1/a0, b2/a0
                a1, a2 = a1/a0, a2/a0

            self.stages.append({'b0': b0, 'b1': b1, 'b2': b2, 'a1': a1, 'a2': a2,
                               'z1': 0.0, 'z2': 0.0})

    def apply(self, x):
        """應用濾波器到單個樣本"""
        y = x
        for s in self.stages:
            out = s['b0'] * y + s['z1']
            s['z1'] = s['b1'] * y - s['a1'] * out + s['z2']
            s['z2'] = s['b2'] * y - s['a2'] * out
            y = out
        return y

    def apply_buffer(self, x):
        """應用濾波器到整個緩衝區"""
        return np.array([self.apply(xi) for xi in x])


def create_bpf_filter():
    """創建帶通濾波器（高通 + 低通）"""
    # 低通濾波器係數
    NUM_LPF = [
        [0.8063260828207, 0, 0],
        [1, -0.3349099821478, 1],
        [0.8764452158503, 0, 0],
        [1, -0.08269016387548, 1],
        [0.8131516681065, 0, 0],
        [1, 0.5521204464881, 1],
        [1.228277124762, 0, 0],
        [1, 1.705652561121, 1],
        [0.00431639855615, 0, 0],
        [1, -0.4218227257396, 1],
        [1, 0, 0],
    ]
    DEN_LPF = [
        [1, 0, 0],
        [1, -0.6719798550872, 0.938845023254],
        [1, 0, 0],
        [1, -0.8264759910073, 0.8561761588872],
        [1, 0, 0],
        [1, -1.10962299915, 0.7141202529829],
        [1, 0, 0],
        [1, -1.413006561919, 0.5638384962434],
        [1, 0, 0],
        [1, -0.6139497794955, 0.9834048810788],
        [1, 0, 0],
    ]

    # 高通濾波器係數
    NUM_HPF = [
        [0.9769037485204, 0, 0],
        [1, -2, 1],
        [0.9424328308459, 0, 0],
        [1, -2, 1],
        [0.9149691441131, 0, 0],
        [1, -2, 1],
        [0.8959987277275, 0, 0],
        [1, -2, 1],
        [0.8863374802187, 0, 0],
        [1, -2, 1],
        [1, 0, 0],
    ]
    DEN_HPF = [
        [1, 0, 0],
        [1, -1.946073828052, 0.9615411660298],
        [1, 0, 0],
        [1, -1.877404882092, 0.8923264412918],
        [1, 0, 0],
        [1, -1.822694925196, 0.837181651256],
        [1, 0, 0],
        [1, -1.78490427193, 0.7990906389804],
        [1, 0, 0],
        [1, -1.765658260281, 0.7796916605933],
        [1, 0, 0],
    ]

    lpf = BiquadFilter(NUM_LPF, DEN_LPF)
    hpf = BiquadFilter(NUM_HPF, DEN_HPF)

    return hpf, lpf


# 從 mseed 文件讀取加速度數據（單位：counts）
acc_x_counts, acc_y_counts, acc_z_counts, sampleRate = read_from_mseed(
    'example.mseed')

if len(acc_x_counts) != len(acc_y_counts) or len(acc_y_counts) != len(acc_z_counts):
    raise ValueError(
        "The lengths of the acceleration data from mseed file are not equal")

# 將 counts 轉換為 gal（counts/10000 = gal）
acc_x = acc_x_counts / 10000
acc_y = acc_y_counts / 10000
acc_z = acc_z_counts / 10000

# 原始加速度用於 LPGM 計算（LPGMCalculator 內部有自己的濾波器）
accel_raw = np.column_stack((acc_x, acc_y, acc_z))

# 濾波後的加速度僅用於波形顯示
if FILTER:
    hpf, lpf = create_bpf_filter()
    acc_x_f = lpf.apply_buffer(hpf.apply_buffer(acc_x))
    acc_y_f = lpf.apply_buffer(hpf.apply_buffer(acc_y))
    acc_z_f = lpf.apply_buffer(hpf.apply_buffer(acc_z))
    accel_display = np.column_stack((acc_x_f, acc_y_f, acc_z_f))
else:
    accel_display = accel_raw

lpgmCalculator = LPGMCalculator(sampleRate)

buffer_size = 30 * sampleRate
accPx = RingBuffer(buffer_size, dtype=np.float64)
accPx.extend(np.zeros(buffer_size))
accPy = RingBuffer(buffer_size, dtype=np.float64)
accPy.extend(np.zeros(buffer_size))
accPz = RingBuffer(buffer_size, dtype=np.float64)
accPz.extend(np.zeros(buffer_size))

t = np.linspace(-29.99, 0, len(accPx))
data_length = len(accel_raw)

# 目標幀率
target_fps = 30
frame_duration = 1.0 / target_fps  # 每幀時間

# 預先建立圖表結構（只建立一次）
plt.ion()  # 開啟互動模式
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.tight_layout(pad=3.0)

labels = ['Acc NS [gal]', 'Acc EW [gal]', 'Acc UD [gal]']
lines = []
for j, ax in enumerate(axes):
    line, = ax.plot(t, np.zeros(len(t)), 'b-')
    lines.append(line)
    ax.set_ylabel(labels[j])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-100, 100)  # 初始範圍，之後會動態調整
    ax.grid(True)
    if j == 2:
        ax.set_xlabel('Time [s]')

title_text = axes[0].set_title('')
fig.canvas.draw()  # 初次繪製

# 儲存背景以便 blitting
backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]

# 預處理 OFFSET 秒的資料（快速跳過，不繪圖）
offset_samples = int(OFFSET * sampleRate)
offset_samples = min(offset_samples, data_length)
for i in range(offset_samples):
    rawAcc = accel_raw[i, :]
    lpgmCalculator.update(rawAcc)
    accPx.append(accel_display[i, 0])
    accPy.append(accel_display[i, 1])
    accPz.append(accel_display[i, 2])

start_time = time.time()
last_draw_time = start_time
current_sample = offset_samples

while current_sample < data_length:
    # 計算當前真實時間對應的樣本位置（從 offset 開始算）
    elapsed_time = time.time() - start_time
    target_sample = offset_samples + int(elapsed_time * sampleRate)
    target_sample = min(target_sample, data_length)

    # 確保至少處理一個樣本（避免空轉）
    if target_sample <= current_sample and current_sample < data_length:
        target_sample = current_sample + 1

    # 處理從 current_sample 到 target_sample 的所有樣本（資料處理不跳過）
    while current_sample < target_sample:
        # 使用原始加速度給 LPGM 計算
        rawAcc = accel_raw[current_sample, :]
        LPGM = lpgmCalculator.update(rawAcc)

        # 使用濾波後的加速度給波形顯示
        accPx.append(accel_display[current_sample, 0])
        accPy.append(accel_display[current_sample, 1])
        accPz.append(accel_display[current_sample, 2])
        current_sample += 1

    # 檢查是否該繪製新幀（限制幀率以確保流暢）
    now = time.time()
    if now - last_draw_time >= frame_duration and current_sample > 0:
        last_draw_time = now

        maxSva30 = lpgmCalculator.getMaxSva30()
        vectPx = accPx[:]
        vectPy = accPy[:]
        vectPz = accPz[:]
        PGA = np.max(np.sqrt(vectPx[-100:]**2 +
                     vectPy[-100:]**2 + vectPz[-100:]**2))
        maxSva = lpgmCalculator.getMaxSva()

        valMax = np.max([np.max(np.abs(vectPx)), np.max(
            np.abs(vectPy)), np.max(np.abs(vectPz))])
        strTitle = 'PGA {:.2f} gal Sva {:.2f} cm/s LPGM {} ({:.2f} cm/s)'.format(
            PGA, np.max(maxSva[:100]), LPGM, maxSva30)

        # 更新資料（不重繪整個圖表）
        data = [vectPx, vectPy, vectPz]
        need_rescale = False

        for j, (line, ax, d) in enumerate(zip(lines, axes, data)):
            line.set_ydata(d)
            # 檢查是否需要調整 Y 軸範圍
            current_ylim = ax.get_ylim()
            new_limit = valMax + 5
            if new_limit > current_ylim[1] * 0.9 or new_limit < current_ylim[1] * 0.3:
                ax.set_ylim(-new_limit, new_limit)
                need_rescale = True

        title_text.set_text(strTitle)

        if need_rescale:
            # Y 軸範圍改變時需要重繪
            fig.canvas.draw()
            backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
        else:
            # 使用 blitting 快速更新
            for j, (ax, line) in enumerate(zip(axes, lines)):
                fig.canvas.restore_region(backgrounds[j])
                ax.draw_artist(line)
                fig.canvas.blit(ax.bbox)
            # 更新標題
            fig.canvas.draw_idle()

        fig.canvas.flush_events()

        # 每秒打印一次狀態
        if int(elapsed_time) != int(elapsed_time - frame_duration):
            print(f"[{elapsed_time:.1f}s] {strTitle}")

    # 短暫休眠避免 CPU 空轉
    time.sleep(0.001)

plt.ioff()
plt.show()
