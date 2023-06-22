import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, freqz
from math import floor, ceil
import os
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
bit_depth = 8  # Глубина битов
voltage_range = 2  # Диапазон напряжения
def bruteforce(lowend,highend,steplow,stephigh):
    # Метод бегущей средней для сглаживания сигнала
    window_size = 10
    # Количество уровней квантования
    quantization_levels = 2 ** bit_depth
     # Шаг квантования
    quantization_step = voltage_range / (quantization_levels - 1)
            
    for lowcut in range(W1 - 10,lowend,steplow):
        for highcut in range(W1 + 100,highend,stephigh):
                
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
                if not os.path.exists(f'sig_{lowcut}-{highcut}.png'):
                    plt.savefig(f'sig_{lowcut}-{highcut}-{order}.png')
                else:
                    continue
                
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
                    line_time = time[0] + i * 0.18
                    plt.axvline(x=line_time, color='r', linestyle='--', linewidth=3)

                plt.tight_layout()
                if not os.path.exists(f'demod_{lowcut}-{highcut}.png'):
                    plt.savefig(f'demod_{lowcut}-{highcut}-{order}.png')
                else:
                    continue
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

                plt.close()
            


if __name__ == "__main__":
    bruteforce(420,3000,5,100)