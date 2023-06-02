import csv
import matplotlib.pyplot as plt
import numpy as np
data = []
W1 = 410
t = np.arange(0, 1,0.0005)
h = 7
Amin = 7 
Amax = 14
m = (Amax - Amin) / (Amax + Amin) # коэф модуляции
#ct=Ac*np.cos(2*math.pi*fc*t)

with open("13.csv") as file:
    reader = csv.reader(file)
    for index,row in enumerate(reader):
        amp = float(row[0])
        data.append(amp)

print(max(data))
data = np.array(data)
plt.plot(t,data)
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда [В]')
plt.show()

