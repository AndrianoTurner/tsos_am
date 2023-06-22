import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.fft import fft,fftfreq


def butter_bandstop(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='bs', output='sos')
        return sos

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandstop(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='bp', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='bp', output='sos')
        return sos

def butter_lowfilt(Wn, fs, order=5):
        sos = butter(order, Wn, analog=False, btype='lowpass', output='sos',fs=fs)
        return sos

def butter_lowfilt_filter(data, Wn, fs, order=5):
        sos = butter_lowfilt(Wn, fs, order=order)
        y = sosfilt(sos, data)
        return y
    
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

fig,axes = plt.subplots(2,sharex=False,sharey=False)
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда [В]')
axes[0].plot(t,data)


filtered = butter_bandstop_filter(data,1000,2000,fs,10)
filtered = butter_lowfilt_filter(data,900,fs,10)
axes[1].plot(filtered)

fig2,axes2 = plt.subplots(2,sharex=False,sharey=False)
xpos = fft(data)[1:len(data)//2]
fpos = np.linspace(0,fs/2,len(xpos))
plt.xlim(0,2500)
axes2[0].plot(fpos,np.abs(xpos)[:999])
xpos = fft(filtered)[1:len(filtered)//2]
fpos = np.linspace(0,fs/2,len(xpos))
axes2[1].plot(fpos,np.abs(xpos)[:999])
plt.show()


