import numpy as np
from dvg_ringbuffer import RingBuffer
import matplotlib.pyplot as plt
from LPGMCalculator import LPGMCalculator
from obspy import read

def read_acceleration_from_sac(file_path):
    stream = read(file_path)
    acceleration_trace = stream[0]
    return acceleration_trace.data

acc_x = read_acceleration_from_sac('X.sac')
acc_y = read_acceleration_from_sac('Y.sac')
acc_z = read_acceleration_from_sac('Z.sac')

if len(acc_x) != len(acc_y) or len(acc_y) != len(acc_z):
    raise ValueError("The lengths of the acceleration data from SAC files are not equal")

accel = np.column_stack((acc_x, acc_y, acc_z))
accel_scaled = accel / 10000
print(accel_scaled)

sampleRate = 20
lpgmCalculator = LPGMCalculator(sampleRate)

accPx = RingBuffer(30*sampleRate, dtype=np.float64)
accPx.extend(np.zeros((30*sampleRate)))
accPy = RingBuffer(30*sampleRate, dtype=np.float64)
accPy.extend(np.zeros((30*sampleRate)))
accPz = RingBuffer(30*sampleRate, dtype=np.float64)
accPz.extend(np.zeros((30*sampleRate)))

t = np.linspace(-29.99, 0, len(accPx))
data_length = len(accel_scaled)
for i in range(data_length):
    rawAcc = accel_scaled[i, :]
    LPGM = lpgmCalculator.update(rawAcc)

    accF = lpgmCalculator.getFilteredAcceleration()
    accPx.append(accF[0])
    accPy.append(accF[1])
    accPz.append(accF[2])

    if i % sampleRate == 0:
        maxSva30 = lpgmCalculator.getMaxSva30()
        vectPx = accPx[:]
        vectPy = accPy[:]
        vectPz = accPz[:]
        PGA = np.max(np.sqrt(vectPx[-100:]**2 + vectPy[-100:]**2 + vectPz[-100:]**2))
        maxSva = lpgmCalculator.getMaxSva()

        valMax = np.max([np.max(np.abs(vectPx)), np.max(np.abs(vectPy)), np.max(np.abs(vectPz))])
        strTitle = 'PGA {:.2f} gal Sva {:.2f} cm/s LPGM {} ({:.2f} cm/s)'.format(
            PGA, np.max(maxSva[:100]), LPGM, maxSva30)

        plt.clf()
        plt.subplot(3, 1, 1)
        plt.title(strTitle, fontname='Meiryo')
        plt.plot(t, vectPx)
        plt.ylim(-valMax-5, valMax+5)
        plt.margins(x=0)
        plt.ylabel('Acc NS [gal]')
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(t, vectPy)
        plt.ylabel('Acc EW [gal]')
        plt.margins(x=0)
        plt.ylim(-valMax-5, valMax+5)
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(t, vectPz)
        plt.ylabel('Acc UD [gal]')
        plt.xlabel('Time [s]')
        plt.margins(x=0)
        plt.ylim(-valMax-5, valMax+5)
        plt.grid()

        print(strTitle)
        plt.pause(0.5)

plt.show()
