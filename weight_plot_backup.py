# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Parameters (NeuroSim 기반)
Pmax = 200  # Maximum Pulse Count
maxNumLevelLTP = 100  # Maximum Number of Conductance Levels for LTP
maxNumLevelLTD = 100  # Maximum Number of Conductance Levels for LTD
Gmax = 0.4 # Maximum Conductance (Weight)
Gmin = -0.4  # Minimum Conductance (Weight)
NL_LTP = 0.5  # LTP Nonlinearity
NL_LTD = -0.5  # LTD Nonlinearity

# Function to compute paramA (NeuroSim 방식)
def get_paramA(NL, numLevel):
    return (1 / abs(NL)) * numLevel

# Compute paramA and paramB for LTP and LTD separately
A_LTP = get_paramA(NL_LTP, maxNumLevelLTP)
A_LTD_base = get_paramA(NL_LTD, maxNumLevelLTD)  # 기본 A_LTD 값
B_LTP = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTP / A_LTP))
#B_LTD_base = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTD / A_LTD_base))  # 기본 B_LTD 값

# NonlinearWeight function (from NeuroSim)
def NonlinearWeight(xPulse, maxNumLevel, A, B, minConductance):
    return B * (1 - np.exp(-xPulse / A)) + minConductance

# Generate Pulse Count (P) values
P_values = np.linspace(0, Pmax, 400)

# Compute LTP using NonlinearWeight function
GLTP = NonlinearWeight(P_values, maxNumLevelLTP, A_LTP, B_LTP, Gmin)

# Compute LTD with Decay effect by modifying A_LTD dynamically
A_LTD = A_LTD_base + 0.8 * (P_values - Pmax/2)
A_LTD = np.maximum(A_LTD, 0.1)  

# Compute B_LTD with a scalar value
B_LTD = (Gmax - Gmin) / (1 - np.exp(-maxNumLevelLTD / np.mean(A_LTD)))

# Compute LTD using modified NonlinearWeight
GLTD = NonlinearWeight(Pmax/2 - P_values, maxNumLevelLTD, A_LTD, B_LTD, Gmax)

# Clip values to ensure they stay within Gmin and Gmax
GLTP = np.clip(GLTP, Gmin, Gmax)
GLTD = np.clip(GLTD, Gmin, Gmax)

# Combine LTP and LTD
G_values = np.where(P_values < Pmax / 2, GLTP, GLTD)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(P_values, G_values, label="Weight Update with LTD Decay", linestyle='-', linewidth=2, color='blue')

# Reference lines
plt.axhline(y=Gmax, color='gray', linestyle=':', label="Gmax")
plt.axhline(y=Gmin, color='gray', linestyle=':', label="Gmin")
plt.axvline(x=Pmax / 2, color='black', linestyle=':', label="LTP to LTD transition")

# Labels and Title
plt.xlabel("Pulse Count (P)")
plt.ylabel("Weight (G) (Conductance)")
plt.title("LTP & LTD Weight Updates with LTD Decay Effect (Modified A_LTD)")
plt.legend()
plt.grid(True)

# Save and Display
plt.savefig('pulse_based_weight_update_with_LTD_decay_paramA.png')
