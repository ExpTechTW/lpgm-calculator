# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:33:50 2021

@author: François Le Neindre / @Nonetype1 

このクラスは、気象庁の長周期地震動階級をリアルタイムの生加速度データから計算するように設計されています。

また、水平成分の絶対速度応答スペクトルを取得することもできます。これは、0.2秒刻みで1.6秒から7.8秒の範囲です。

使用した方法は、[Nigam, Navin C. and Jennings, Paul C. (1969) Calculation of 
response spectra from strong-motion earthquake records.]からのものです。

/！\読み取り値は、加速度計が建物の1階、または地面に接続された最も水平な面に配置されている場合に
のみ有効です。
サンプルレートは、時間の経過とともに一定である必要があります。
加速度入力はフィルタリングされていない必要があります（重力補正なし）


This class is designed to compute the Japan Meteorological Agency Long-Period 
Ground Motion class, from real-time raw acceleration data.
It is also possible to retrieve the Absolute Velocity Response Spectrum of 
horizontal components, spanning between 1.6s and 7.8s with 0.2s increments.

The method used is from [Nigam, Navin C. and Jennings, Paul C. (1969) 
Calculation of response spectra from strong-motion earthquake records.]

Note: The readings are only valid if the accelerometer is placed on the first 
floor of any building or on a solid surface tied to the ground as level as 
possible. The sample rate has to be constant over time.
The acceleration input has to be unfiltered (ie. without gravity compensation)

"""

import numpy as np
from dvg_ringbuffer import RingBuffer
from scipy import signal


class LPGMCalculator:

    def __init__(self, sampleRate):
        self.sampleRate = sampleRate
        self.sampleTime = 1.0 / sampleRate

        # High-Pass Filter (cut-off at 20 seconds)
        baf = signal.butter(2, 0.05, 'hp', fs=sampleRate, output='ba')
        self.bf, self.af = baf[0], baf[1]

        # Sva periods array (1.6s to 7.8s with 0.2s increments)
        self.Np = 32
        self.periods = np.linspace(1.6, 7.8, self.Np)
        self.Sva = np.zeros(self.Np)

        beta = 0.05  # Damping factor

        # Initialize A and B matrices
        self.A = np.zeros((2, 2, self.Np))
        self.B = np.zeros((2, 2, self.Np))
        self.xi = np.zeros((2, 2, self.Np))
        self.init = True

        # Calculate A and B matrices for each period
        for j in range(self.Np):
            w = 2 * np.pi / self.periods[j]
            sqrt_term = np.sqrt(1 - beta**2)
            SW = np.sin(w * sqrt_term * self.sampleTime)
            CW = np.cos(w * sqrt_term * self.sampleTime)
            E = np.exp(-beta * w * self.sampleTime)

            beta_sqrt = beta / sqrt_term
            w_sqrt = w * sqrt_term

            # A matrix
            self.A[0, 0, j] = E * (beta_sqrt * SW + CW)
            self.A[0, 1, j] = SW * E / w_sqrt
            self.A[1, 0, j] = -SW * E * w / sqrt_term
            self.A[1, 1, j] = E * (-beta_sqrt * SW + CW)

            # B matrix (simplified calculations)
            w2 = w**2
            w3 = w2 * w
            dt = self.sampleTime
            term1 = (2*beta**2 - 1) / (w2 * dt) + beta / w
            term2 = 2*beta / (w3 * dt) + 1 / w2

            self.B[0, 0, j] = E * \
                (term1 * SW / w_sqrt + term2 * CW) - 2*beta / (w3 * dt)
            self.B[0, 1, j] = -E * ((2*beta**2 - 1) / (w2 * dt) * SW / w_sqrt +
                                    2*beta / (w3 * dt) * CW) + 2*beta / (w3 * dt) - 1 / w2
            self.B[1, 0, j] = E * (term1 * (CW - beta*SW/sqrt_term) -
                                   term2 * (w_sqrt*SW + beta*w*CW)) + 1 / (w2 * dt)
            self.B[1, 1, j] = -E * ((2*beta**2 - 1) / (w2 * dt) * (CW - beta*SW/sqrt_term) -
                                    2*beta / (w3 * dt) * (w_sqrt*SW + beta*w*CW)) - 1 / (w2 * dt)

        # Initialize buffers
        self.accH = self.accH_1 = self.accH_2 = np.zeros(3)
        self.accHF = self.accHF_1 = np.zeros(3)
        self.vel = self.acc0 = np.zeros(3)
        self.LPGM = 0

        buffer_size = 30 * sampleRate
        self.maxSva = RingBuffer(buffer_size, dtype=np.float64)
        self.maxSva.extend(np.zeros(buffer_size))

    def update(self, rawAcceleration):
        """
        Update LPGM, filtered 3D acceleration, 3D velocity
        Input: raw acceleration components np.array([ax, ay, az])
               First 2 components must be horizontal
        Output: LPGM value from max Sva of last 30 seconds
        """

        if self.init:
            self.acc0 = rawAcceleration
            self.init = False

        # Shift filter registers
        self.accH_2, self.accH_1 = self.accH_1, self.accH
        self.accH = rawAcceleration - self.acc0

        # Filter acceleration
        accHFi = (self.bf[0]*self.accH + self.bf[1]*self.accH_1 + self.bf[2]*self.accH_2 -
                  self.af[1]*self.accHF - self.af[2]*self.accHF_1)
        self.accHF_1, self.accHF = self.accHF, accHFi

        # Compute velocity
        self.vel += (self.accHF_1 + self.accHF) * self.sampleTime / 2

        # Compute Sva for each period
        for j in range(self.Np):
            acc_input = np.array([self.accHF_1[:2], self.accHF[:2]])
            self.xi[:, :, j] = (np.matmul(self.A[:, :, j], self.xi[:, :, j]) +
                                np.matmul(self.B[:, :, j], acc_input))
            self.Sva[j] = np.sqrt((self.xi[1, 0, j] + self.vel[0])**2 +
                                  (self.xi[1, 1, j] + self.vel[1])**2)

        # Update LPGM class
        self.maxSva.appendleft(np.max(self.Sva))
        self.maxSva30 = np.max(self.maxSva)

        if self.maxSva30 < 5:
            self.LPGM = 0
        elif self.maxSva30 < 15:
            self.LPGM = 1
        elif self.maxSva30 < 50:
            self.LPGM = 2
        elif self.maxSva30 < 100:
            self.LPGM = 3
        else:
            self.LPGM = 4
        return self.LPGM

    def getSva(self):
        """Returns Absolute Response Velocity Spectrum from 1.6 to 7.8s with 0.2s increment"""
        return self.Sva

    def getFilteredAcceleration(self):
        """Returns High-Pass filtered acceleration (recommended for PGA computation)"""
        return self.accHF

    def getVelocity(self):
        """Returns current 3D velocity"""
        return self.vel

    def getMaxSva30(self):
        """Returns maximum Sva value of the last 30 seconds"""
        return self.maxSva30

    def getMaxSva(self):
        """Returns current maximum Sva value"""
        return self.maxSva
