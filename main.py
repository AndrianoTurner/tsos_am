import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, freqz
from math import floor, ceil
import os
import multiprocessing
import itertools
# Чтение данных из файла CSV
df = pd.read_csv('13.csv', header=None)
amplitude = df[0].values.astype(float)
print(amplitude)
time = np.linspace(0, 1, len(amplitude))  # Временные метки
W1 = 410
# Параметры фильтрации
fs = 41000 # Частота дискретизации
lowcut = 410  # Нижняя граница частотного диапазона
highcut = 2500  # Верхняя граница частотного диапазона
order = 5  # Порядок фильтра
# Параметры АЦП
bit_depth = 10  # Глубина битов
voltage_range = 14  # Диапазон напряжения
def bruteforce(args):
    print(args)
    lowcut = 400
    highcut = args[0]
    order = args[1]
    voltage_range = args[2]
    if os.path.exists(f'./images/demod_{lowcut}-{highcut}-{order}-{voltage_range}.png') or os.path.exists(f'./images/sig_{lowcut}-{highcut}-{order}-{voltage_range}.png'):
        return
    # Метод бегущей средней для сглаживания сигнала
    window_size = 10
    # Количество уровней квантования
    quantization_levels = 2 ** bit_depth
     # Шаг квантования
    quantization_step = voltage_range / (quantization_levels - 1)
            
    
    # Проверка на корректность нормализованных частот
    if lowcut >= fs or highcut >= fs:
        raise ValueError("Некорректные значения границ частотного диапазона. Убедитесь, что fs правильно задана.")

                # Нормализация границ частотного диапазона
    lowcut_normalized = lowcut / (fs / 2)
    highcut_normalized = highcut / (fs / 2)
    # Расчет коэффициентов фильтра
    b, a = butter(order, [lowcut_normalized, highcut_normalized], btype='band')

                
    smoothed_signal = np.convolve(amplitude, np.ones(window_size)/window_size, mode='same')

    # Применение фильтра к сглаженному сигналу
    filtered_signal = filtfilt(b, a, smoothed_signal)
    
    # Демодуляция АМ-сигнала
    demodulated_signal = np.abs(filtered_signal)

                # Масштабирование демодулированного сигнала
    scaled_signal = demodulated_signal * voltage_range

                

                # Квантование сигнала с округлением
    quantized_signal = np.round((scaled_signal / quantization_step) + 0.5) * quantization_step
    quantized_signal = np.round((quantized_signal - 1) / 2) * 2 + 1

    # Построение графиков
    plt.figure(figsize=(10, 18))

    plt.subplot(3, 1, 1)
    plt.plot(time, amplitude)
    plt.grid(True)
    plt.title('Исходный сигнал')

    plt.subplot(3, 1, 2)
    plt.plot(time, smoothed_signal)
    plt.grid(True)
    plt.title('Сглаженный сигнал')

    plt.subplot(3, 1, 3)
    plt.plot(time, filtered_signal)
    plt.grid(True)
    plt.title('Отфильтрованный сигнал')
    
    plt.tight_layout()
    plt.savefig(f'./images/sig_{lowcut}-{highcut}-{order}-{voltage_range}.png')
    plt.close()
                
                # Создание второго окна с графиками демодулированного и квантованного сигналов
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time, demodulated_signal)
    ax1.grid(True)
    ax1.set_title('Демодулированный сигнал')
    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Амплитуда')

    ax2.bar(time, quantized_signal, width=1/(fs * 0.1))
    ax2.grid(True)
    ax2.set_title('Квантованный сигнал')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Амплитуда')

    for i in range(5):
        line_time = time[0] + i * 0.16
        plt.axvline(x=line_time, color='r', linestyle='--', linewidth=3)

    plt.tight_layout()
    plt.savefig(f'./images/demod_{lowcut}-{highcut}-{order}-{voltage_range}.png')
    plt.close()
                # Создание третьего окна с графиками характеристик фильтра
                #w, h = freqz(b, a, fs=fs)

                #plt.figure(figsize=(10, 8))
                #plt.subplot(2, 1, 1)
                #plt.plot(w, 20 * np.log10(abs(h)))
                #plt.grid(True)
                #plt.title('Амплитудно-частотная характеристика фильтра')
                #plt.ylabel('Уровень (дБ)')
                #plt.xlabel('Частота (Гц)')

            # plt.subplot(2, 1, 2)
                #plt.plot(w, np.unwrap(np.angle(h)))
                #plt.grid(True)
                #plt.title('Фазочастотная характеристика фильтра')
                #plt.ylabel('Фаза (рад)')
                #plt.xlabel('Частота (Гц)')
                #plt.tight_layout()


def create_filter(fs,lowcut,highcut,order,bit_depth,voltage_range):
    df = pd.read_csv('13.csv', header=None)
    amplitude = df[0].values.astype(float)
    print(amplitude)
    time = np.linspace(0, 1, len(amplitude))  # Временные метки

    # Проверка на корректность нормализованных частот
    if lowcut >= fs or highcut >= fs:
        raise ValueError("Некорректные значения границ частотного диапазона. Убедитесь, что fs правильно задана.")

    # Нормализация границ частотного диапазона
    lowcut_normalized = lowcut / (fs / 2)
    highcut_normalized = highcut / (fs / 2)

    # Расчет коэффициентов фильтра
    b, a = butter(order, [lowcut_normalized, highcut_normalized], btype='band')

    # Метод бегущей средней для сглаживания сигнала
    window_size = 10
    smoothed_signal = np.convolve(amplitude, np.ones(window_size)/window_size, mode='same')

    # Применение фильтра к сглаженному сигналу
    filtered_signal = filtfilt(b, a, smoothed_signal)
    # Демодуляция АМ-сигнала
    demodulated_signal = np.abs(filtered_signal)

    # Масштабирование демодулированного сигнала
    scaled_signal = demodulated_signal * voltage_range

    # Количество уровней квантования
    quantization_levels = 2 ** bit_depth

    # Шаг квантования
    quantization_step = voltage_range / (quantization_levels - 1)

    # Квантование сигнала с округлением
    quantized_signal = np.round((scaled_signal / quantization_step) + 0.5) * quantization_step
    quantized_signal = np.round((quantized_signal - 1) / 2) * 2 + 1

    # Построение графиков
    plt.figure(figsize=(10, 18))

    plt.subplot(3, 1, 1)
    plt.plot(time, amplitude)
    plt.grid(True)
    plt.title('Исходный сигнал')

    plt.subplot(3, 1, 2)
    plt.plot(time, smoothed_signal)
    plt.grid(True)
    plt.title('Сглаженный сигнал')

    plt.subplot(3, 1, 3)
    plt.plot(time, filtered_signal)
    plt.grid(True)
    plt.title('Отфильтрованный сигнал')


    plt.tight_layout()

    # Создание второго окна с графиками демодулированного и квантованного сигналов
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time, demodulated_signal)
    ax1.grid(True)
    ax1.set_title('Демодулированный сигнал')
    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Амплитуда')

    ax2.bar(time, quantized_signal, width=1/(fs * 0.1))
    ax2.grid(True)
    ax2.set_title('Квантованный сигнал')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Амплитуда')

    for i in range(8):
        line_time = time[0] + i * 0.125
        plt.axvline(x=line_time, color='r', linestyle='--', linewidth=3)

    plt.tight_layout()

    # Создание третьего окна с графиками характеристик фильтра
    w, h = freqz(b, a, fs=fs)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.grid(True)
    plt.title('Амплитудно-частотная характеристика фильтра')
    plt.ylabel('Уровень (дБ)')
    plt.xlabel('Частота (Гц)')

    plt.subplot(2, 1, 2)
    plt.plot(w, np.unwrap(np.angle(h)))
    plt.grid(True)
    plt.title('Фазочастотная характеристика фильтра')
    plt.ylabel('Фаза (рад)')
    plt.xlabel('Частота (Гц)')

    plt.tight_layout()

    # Открытие всех окон программы
    plt.show()

            
def pool_run():
    pool = multiprocessing.Pool()
    input = itertools.product(range(1000,2000,150),range(5,10),range(1,7))
    pool.map(bruteforce,input)

if __name__ == "__main__":
    create_filter(35000,400,2000,5,2,1)
    #pool_run()
    
    