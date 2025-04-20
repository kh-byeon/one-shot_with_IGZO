# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters (NeuroSim 기반)
Pmax = 200  # Maximum Pulse Count
maxNumLevelLTP = 100  # Maximum Number of Conductance Levels for LTP
maxNumLevelLTD = 100  # Maximum Number of Conductance Levels for LTD
Gmax = 1  # Maximum Conductance (Weight)
Gmin = 0  # Minimum Conductance (Weight)

# Sweep할 NL 값 설정
NL_LTP_values = [0.01, 1, 2, 3, 4, 5, 6]  # LTP 비선형성 값
NL_LTD_values = [-0.01, -1, -2, -3, -4, -5, -6]  # LTD 비선형성 값

# Step data
step_data = np.array([
    0.002919708, 0.018978102, 0.020437956, 0.030656934, 0.042335766, 0.042335766, 0.064233577, 0.080291971, 0.090510949, 0.094890511,
0.116788321, 0.131386861, 0.143065693, 0.15620438, 0.17080292, 0.186861314, 0.201459854, 0.216058394, 0.227737226, 0.242335766,
0.256934307, 0.271532847, 0.287591241, 0.299270073, 0.306569343, 0.327007299, 0.340145985, 0.354744526, 0.362043796, 0.37810219,
0.39270073, 0.404379562, 0.420437956, 0.43649635, 0.446715328, 0.458394161, 0.470072993, 0.483211679, 0.496350365, 0.508029197,
0.518248175, 0.528467153, 0.541605839, 0.554744526, 0.564963504, 0.57810219, 0.58540146, 0.595620438, 0.605839416, 0.61459854,
0.624817518, 0.639416058, 0.648175182, 0.654014599, 0.664233577, 0.681751825, 0.689051095, 0.694890511, 0.700729927, 0.713868613,
0.722627737, 0.731386861, 0.740145985, 0.747445255, 0.759124088, 0.76350365, 0.773722628, 0.783941606, 0.794160584, 0.797080292,
0.805839416, 0.811678832, 0.82189781, 0.82919708, 0.837956204, 0.84379562, 0.851094891, 0.856934307, 0.862773723, 0.874452555,
0.881751825, 0.887591241, 0.893430657, 0.896350365, 0.909489051, 0.913868613, 0.922627737, 0.927007299, 0.934306569, 0.938686131,
0.945985401, 0.948905109, 0.962043796, 0.964963504, 0.972262774, 0.976642336, 0.982481752, 0.986861314, 0.994160584, 1,
0.986861314, 0.973722628, 0.960583942, 0.947445255, 0.931386861, 0.919708029, 0.909489051, 0.893430657, 0.881751825, 0.861313869,
0.851094891, 0.840875912, 0.82919708, 0.816058394, 0.802919708, 0.794160584, 0.77810219, 0.764963504, 0.754744526, 0.740145985,
0.727007299, 0.715328467, 0.705109489, 0.691970803, 0.678832117, 0.665693431, 0.658394161, 0.645255474, 0.633576642, 0.620437956,
0.605839416, 0.597080292, 0.58540146, 0.573722628, 0.564963504, 0.553284672, 0.538686131, 0.529927007, 0.516788321, 0.508029197,
0.496350365, 0.490510949, 0.475912409, 0.465693431, 0.455474453, 0.44379562, 0.43649635, 0.42189781, 0.411678832, 0.402919708,
0.39270073, 0.383941606, 0.37080292, 0.367883212, 0.353284672, 0.340145985, 0.338686131, 0.325547445, 0.315328467, 0.306569343,
0.297810219, 0.287591241, 0.277372263, 0.271532847, 0.262773723, 0.254014599, 0.237956204, 0.237956204, 0.230656934, 0.223357664,
0.216058394, 0.208759124, 0.197080292, 0.19270073, 0.182481752, 0.175182482, 0.166423358, 0.162043796, 0.159124088, 0.147445255,
0.144525547, 0.138686131, 0.132846715, 0.131386861, 0.119708029, 0.109489051, 0.112408759, 0.105109489, 0.099270073, 0.093430657,
0.089051095, 0.083211679, 0.083211679, 0.071532847, 0.070072993, 0.065693431, 0.061313869, 0.058394161, 0.055474453, 0.055474453

])
step_normalized = (step_data - np.min(step_data)) / (np.max(step_data) - np.min(step_data))

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
#colors_LTP = cm.Blues(np.linspace(0.3, 1, len(NL_LTP_values)))
#colors_LTD = cm.Reds(np.linspace(0.3, 1, len(NL_LTD_values)))
colors_LTP = cm.jet(np.linspace(0.7, 0.3, len(NL_LTP_values)))  # LTP: 파란색 → 빨간색 (밝은색)
colors_LTD = cm.jet(np.linspace(0.7, 0.3, len(NL_LTP_values)))  # LTP: 파란색 → 빨간색 (밝은색)
#colors_LTD = cm.jet(np.linspace(0, 0.7, len(NL_LTD_values)))  # LTD: 파란색 → 초록색 (어두운색)

# LTP 곡선 (X축을 10배 증가)
for i, NL_LTP in enumerate(NL_LTP_values):
    A_LTP = get_paramA(NL_LTP, maxNumLevelLTP)
    B_LTP = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTP / A_LTP))
    GLTP = NonlinearWeight(P_values, maxNumLevelLTP, A_LTP, B_LTP, Gmin)
    GLTP = np.clip(GLTP, Gmin, Gmax)
    linestyle = 'k--' if NL_LTP == 0.01 else '-'
    plt.plot(P_values[:200] * 10, GLTP[:200], linestyle, color=colors_LTP[i] if NL_LTP != 0.01 else 'k',
             linewidth=4, alpha=1, label=f"LTP NL={NL_LTP}")

# LTD 곡선 (X축을 10배 증가)
for i, NL_LTD in enumerate(NL_LTD_values):
    A_LTD = get_paramA(NL_LTD, maxNumLevelLTD)
    B_LTD = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTD / A_LTD))
    GLTD = NonlinearWeight(P_values - Pmax / 2, maxNumLevelLTD, A_LTD, -B_LTD, Gmax)
    GLTD = np.clip(GLTD, Gmin, Gmax)
    linestyle = 'k--' if NL_LTD == -0.01 else '-'
    plt.plot(P_values[200:] * 10, GLTD[200:], linestyle, color=colors_LTD[i] if NL_LTD != -0.01 else 'k',
             linewidth=4, alpha=1, label=f"LTD NL={NL_LTD}")

#  Step Data를 Scatter로 표시 (X축도 10배 증가)
# 1~100 구간 (파란색)
plt.scatter(P_values[:100] * 20, step_normalized[:100], color='blue', s=20, alpha=1, label="Step Data (1-100)", zorder=3)

# 101~200 구간 (빨간색)
plt.scatter(P_values[100:200] * 20, step_normalized[100:200], color='red', s=20, alpha=1, label="Step Data (101-200)", zorder=3)

#  X축 범위 0~2000으로 설정
plt.xlim(0, 2000)

#  X축 눈금 설정 (500 간격으로 정렬)
plt.xticks(np.arange(0, 2100, 200), fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')

#  Tick을 안쪽으로 설정
plt.tick_params(axis="both", direction="in", length=6, width=2)

#  Grid 설정
plt.grid(True, linestyle='-', linewidth=1.0, alpha=0.5)

# X축, Y축 테두리 두껍게 설정
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.0)

# Labels
#plt.xlabel("Pulse Count", fontsize=18, fontweight='bold')
#plt.ylabel("Normalized Weight", fontsize=18, fontweight='bold')

# 저장 및 출력
plt.savefig("LTP_LTD_NL_sweep_with_scatter_fixed.png", dpi=300)
plt.show()
