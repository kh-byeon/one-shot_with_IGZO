# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Pmax = 200
maxNumLevelLTP = 100
maxNumLevelLTD = 100
Gmax = 1
Gmin = 0.1

# Sweep할 NL 값 설정
NL_LTP_values = [0.1, 0.5, 1, 2, 5]  # LTP 비선형성 값
NL_LTD_values = [-0.1, -0.5, -1, -2, -5]  # LTD 비선형성 값

# Pulse Count (P) 설정
P_values = np.linspace(0, Pmax, 400)

# Nonlinear Weight function
def NonlinearWeight(xPulse, A, B, minConductance):
    return B * (1 - np.exp(-xPulse / A)) + minConductance

# Function to compute paramA
def get_paramA(NL, numLevel):
    return (1 / abs(NL)) * numLevel

# 그래프 설정
plt.figure(figsize=(10, 7))

# Color map 설정
colors_LTP = plt.cm.Blues(np.linspace(0.3, 1, len(NL_LTP_values)))
colors_LTD = plt.cm.Reds(np.linspace(0.3, 1, len(NL_LTD_values)))

# 여러 NL 값에 대한 LUT 곡선 플로팅
for i, (NL_LTP, NL_LTD) in enumerate(zip(NL_LTP_values, NL_LTD_values)):
    A_LTP = get_paramA(NL_LTP, maxNumLevelLTP)
    A_LTD = get_paramA(NL_LTD, maxNumLevelLTD)

    B_LTP = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTP / A_LTP))
    B_LTD = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTD / A_LTD))

    LUT_LTP = NonlinearWeight(P_values[:200], A_LTP, B_LTP, Gmin)
    LUT_LTD = NonlinearWeight(P_values[:200], A_LTD, -B_LTD, Gmax)

    plt.plot(P_values[:200], LUT_LTP, linestyle='-', linewidth=2, color=colors_LTP[i], label=f"LTP NL={NL_LTP}")
    plt.plot(P_values[:200], LUT_LTD, linestyle='-', linewidth=2, color=colors_LTD[i], label=f"LTD NL={NL_LTD}")

# Labels and title
plt.xlabel("Pulse Count (P)", fontsize=14, fontweight='bold')
plt.ylabel("Weight (G)", fontsize=14, fontweight='bold')
plt.title("LTP & LTD LUT for Different NL Values", fontsize=16, fontweight='bold')

# Legend and grid
plt.legend()
plt.grid(True)

# Save and show plot
plt.savefig('LTP_LTD_NL_sweep.png', dpi=300)
