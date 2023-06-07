import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
data = []
W1 = 410
t = np.arange(0, 1,0.0005)
h = 7
Amin = 7 
Amax = 14
fs = 41000
m = (Amax - Amin) / (Amax + Amin) # коэф модуляции
ct=Amin*np.cos(2*math.pi*W1*t)

with open("13.csv") as file:
    reader = csv.reader(file)
    for index,row in enumerate(reader):
        amp = float(row[0])
        data.append(amp)

print(max(data))
data = np.array(data)

plt.xlabel('Время [с]')
plt.ylabel('Амплитуда [В]')
plt.plot(t,data)
plt.figure()

order,wn = signal.buttord([700,1000], [600,1100], 3, 40,False,fs)
sos = signal.butter(order, wn , 'bp',fs=fs, output='sos')
filtered = signal.sosfilt(sos, data)
plt.plot(t,filtered)
plt.show()


