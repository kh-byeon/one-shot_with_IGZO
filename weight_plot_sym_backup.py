# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.font_manager as fm


# Parameters (NeuroSim 기반)
Pmax = 200  # Maximum Pulse Count
maxNumLevelLTP = 100  # Maximum Number of Conductance Levels for LTP
maxNumLevelLTD = 100  # Maximum Number of Conductance Levels for LTD
Gmax = 1  # Maximum Conductance (Weight)
Gmin = 0  # Minimum Conductance (Weight)

# Sweep할 NL 값 설정
NL_LTP_values = [0.01,1, 2, 3, 4, 5, 6]  # LTP 비선형성 값
NL_LTD_values = [-0.01,-1, -2, -3, -4, -5, -6]  # LTD 비선형성 값

# Pulse Count (P) 설정
P_values = np.linspace(0, Pmax, 400)

# NonlinearWeight function (from NeuroSim)
def NonlinearWeight(xPulse, maxNumLevel, A, B, minConductance):
    return B * (1 - np.exp(-xPulse / A)) + minConductance


# Function to compute paramA (NeuroSim 방식)
def get_paramA(NL, numLevel):
    return (1 / abs(NL)) * numLevel

# 기본 weight update 그래프
plt.figure(figsize=(8, 6))

# Color map 설정
colors_LTP = cm.Blues(np.linspace(0.3, 1, len(NL_LTP_values)))
colors_LTD = cm.Reds(np.linspace(0.3, 1, len(NL_LTD_values)))

# NL sweep 추가 (여러 곡선)
for i, NL_LTP in enumerate(NL_LTP_values):
    A_LTP = get_paramA(NL_LTP, maxNumLevelLTP)
    B_LTP = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTP / A_LTP))
    GLTP = NonlinearWeight(P_values, maxNumLevelLTP, A_LTP, B_LTP, Gmin)
    GLTP = np.clip(GLTP, Gmin, Gmax)
    linestyle = 'k--' if NL_LTP == 0.01 else '-'
    plt.plot(P_values[:200], GLTP[:200], linestyle, color=colors_LTP[i] if NL_LTP != 0.01 else 'k', linewidth=5, alpha=0.8, label=f"LTP NL={NL_LTP}")

for i, NL_LTD in enumerate(NL_LTD_values):
    A_LTD_base = get_paramA(NL_LTD, maxNumLevelLTD)
    A_LTD = A_LTD_base
    B_LTD = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTD / A_LTD))
    GLTD = NonlinearWeight(P_values - Pmax/2, maxNumLevelLTD, A_LTD, -B_LTD, Gmax)
    GLTD = np.clip(GLTD, Gmin, Gmax)
    linestyle = 'k--' if NL_LTD == -0.01 else '-'
    plt.plot(P_values[200:], GLTD[200:], linestyle, color=colors_LTD[i] if NL_LTD != -0.01 else 'k', linewidth=5, alpha=0.8, label=f"LTD NL={NL_LTD}")


# Reference Lines
#plt.axhline(y=Gmax, color='gray', linestyle=':', label="Gmax")
#plt.axhline(y=Gmin, color='gray', linestyle=':', label="Gmin")
#plt.axvline(x=Pmax / 2, color='black', linestyle=':', label="LTP to LTD transition")


# x축, y축 눈금의 글자를 굵게 설정
plt.xticks(fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')

# Labels and Title
plt.xlabel("Pulse Count", fontsize=18, fontweight='bold')
plt.ylabel("Normalized Weight", fontsize=18, fontweight='bold')
#plt.title("LTP & LTD Weight Updates with NL Sweep")
#plt.legend(loc="lower right", fontsize=8)
plt.grid(True)

# Save and Show Plot
plt.savefig("LTP_LTD_NL_sweep1.png", dpi=300)
