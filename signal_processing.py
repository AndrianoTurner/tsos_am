import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
from scipy.fft import fft,fftfreq
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

order,wn = signal.buttord([1000,1500], [900,1600], 3, 40,False,fs/2)
sos = signal.butter(order, wn , 'bs',fs=fs, output='sos')
filtered = signal.sosfilt(sos, data)
plt.plot(t,filtered)

plt.figure()

xpos = fft(data)[1:len(data)//2]
fpos = np.linspace(0,fs/2,len(xpos))
plt.xlim(0,2500)
plt.plot(fpos,np.abs(xpos)[:999])
plt.show()


