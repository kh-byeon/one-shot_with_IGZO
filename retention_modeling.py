import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


retention_time = np.loadtxt("./retention_rawdata/WF9_40_20_1_40_20_1_Die6_retention_Data.csv", delimiter=",", usecols=0, skiprows=1)
storage_voltage = np.loadtxt("./retention_rawdata/WF9_40_20_1_40_20_1_Die6_retention_Data.csv", delimiter=",", usecols=1, skiprows=1)


def exponential_decay(t, A, tau, C):
    """
    t: retention time
    A: initial amplitude
    tau: decay constant  736.7138640826427
    C: offset
    """
    return A * np.exp(-t / tau) + C


initial_guess = [1.0, 100.0, 0.0] 
params, covariance = curve_fit(exponential_decay, retention_time, storage_voltage, p0=initial_guess)


A, tau, C = params
print(f"Estimated Parameters:\n A = {A}\n tau (Decay Constant) = {tau}\n C = {C}")

plt.figure(figsize=(10, 6))
plt.scatter(retention_time, storage_voltage, label="Measured Data", color="blue", s=10)
plt.plot(retention_time, exponential_decay(retention_time, *params), label="Fitted Curve", color="red")
plt.xlabel("Retention Time (s)")
plt.ylabel("Storage Node Voltage (V)")
plt.title("Exponential Decay Fitting")
plt.legend()
plt.grid()
plt.savefig('ret_model.png')
plt.show()
