import math
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

# Функция гауссового импульса
def gauss(q, m, d_g, w_g, d_t, eps=1, mu=1, Sc=1):
    return np.exp(-((((q - m*np.sqrt(eps*mu)/Sc)
                      - (d_g / d_t)) / (w_g / dt)) ** 2))

# Функция - дискретизатор
def sampler(obj, d_obj: float) -> int:
    return math.floor(obj / d_obj + 0.5)

# Параметры моделирования

W0 = 120.0 * np.pi # Волновое сопротивление свободного пространства

Sc = 1.0 # Число Куранта

c = 299792458.0 # Скорость света


maxSize_m = 0.9 # Область моделирования

dx = 5e-3 # Дискрет по пространству

maxSize = math.floor(maxSize_m / dx + 0.5) # Размер облести моделирования в отсчетах
# Параметры слоев


d0_m = 0.2 # До начала слоев

d1_m = 0.15 # Толщина первого слоя

d2_m = 0.35 # Толщина второго слоя

d3_m = 0.12 # Толщина третьего слоя


posL1min = sampler(d0_m, dx) # Отсчет начала слоёв

posL2min = sampler(d0_m + d1_m, dx) # Отсчет начала первого слоя

posL3min = sampler(d0_m + d1_m + d2_m, dx) # Отсчет начала второго слоя

posL4min = sampler(d0_m + d1_m + d2_m + d3_m, dx) # Отсчёт начала третьего слоя

# Параметры среды

# Диэлектрические проницаемости
eps = np.ones(maxSize)
eps[posL1min:posL2min] = 4.7
eps[posL2min:posL3min] = 2.2
eps[posL3min:posL4min] = 3.2
eps[posL4min:] = 1.0


mu = 1.0 # Магнитная проничаемость

maxTime_s = 100e-9 # Время расчёта в секундах

dt = Sc * dx / c # Расчёт дискрета по времени

maxTime = sampler(maxTime_s, dt) # Расчёт времени

tlist = np.arange(0, maxTime * dt, dt) # Временная сетка

df = 1.0 / (maxTime * dt) # Шаг по частоте

flist = np.arange(-maxTime / 2 * df, maxTime / 2 * df, df) # Частотная сетка

# Параметры гауссова сигнала

A_0 = 100 # Уровень ослабления в начале

A_max = 100 # Уровень ослабления на частоте F_max

F_max = 3e9 # Ширина спектра по уровню 0.01

wg = np.sqrt(np.log(A_max)) / (np.pi * F_max) # Параметр длины импульса

dg = wg * np.sqrt(np.log(A_0)) # Параметр остановки импульса

sourcePos_m = 0.15 # Положение источника

sourcePos = math.floor(sourcePos_m / dx + 0.5) # Положение источника в расчётах

probe1Pos_m = 0.02 # Обозначим положение датчика

probe1Pos = sampler(probe1Pos_m, dx) # Положение датчика в расчётах

probe1Ez = np.zeros(maxTime) # Инициализация датчика

# Инициализация полей
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize - 1)


Ez0 = np.zeros(maxTime) # Массив, содержащий падающий сигнал

# Вспомогательные коэффициенты

# Для расчёта граничных условий
Sc1 = Sc / np.sqrt(mu * eps)
k1 = -1 / (1 / Sc1 + 2 + Sc1)
k2 = 1 / Sc1 - 2 + Sc1
k3 = 2 * (Sc1 - 1 / Sc1)
k4 = 4 * (1 / Sc1 + Sc1)
# Хранение полей за предыдущее время, слева
oldEzL1 = np.zeros(3)
oldEzL2 = np.zeros(3)

# Хранение полей за предыдущее время, справа
oldEzR1 = np.zeros(3)
oldEzR2 = np.zeros(3)

# Расчёт полей
for t in range(1, maxTime):
 # Расчёт компонента поля H
     Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)
 # Источник возбуждения Hy
     Hy[sourcePos - 1] -= (Sc / W0) * \
         gauss(t, sourcePos, dg, wg, dt,
               eps=eps[sourcePos], mu=mu)
 # Расчет компоненты поля E
     Ez[1:-1] = Ez[1: -1] + \
        (Hy[1:] - Hy[: -1]) * Sc * W0 / eps[1: -1]
 # Источник возбуждения Ez
     Ez0[t] = Sc * gauss(t + 1, sourcePos, dg, wg, dt,
                         eps=eps[sourcePos], mu=mu)
     Ez[sourcePos] += Ez0[t]
 # Граничные условия для поля E cлева
     Ez[0] = (k1[0] * (k2[0] * (Ez[2] + oldEzL2[0]) +
                       k3[0] * (oldEzL1[0] + oldEzL1[2] -
                                Ez[1] - oldEzL2[1]) -
                       k4[0] * oldEzL1[1]) - oldEzL2[2])
     oldEzL2[:] = oldEzL1[:]
     oldEzL1[:] = Ez[0: 3]
 # Граничные условия для поля E cправа
     Ez[-1] = (k1[-1] * (k2[-1] * (Ez[-3] + oldEzR2[-1]) +
                         k3[-1] * (oldEzR1[-1] + oldEzR1[-3] -
                                   Ez[-2] - oldEzR2[-2]) -
                         k4[-1] * oldEzR1[-2]) - oldEzR2[-3])
     oldEzR2[:] = oldEzR1[:]
     oldEzR1[:] = Ez[-3:]
     

     probe1Ez[t] = Ez[probe1Pos]  # Регистрация поля в датчике
     
# Расчет спектра зарегистрированного сигнала
Ez1Spec = fftshift(np.abs(fft(probe1Ez)))
Ez0Spec = fftshift(np.abs(fft(Ez0)))
Gamma = Ez1Spec / Ez0Spec


fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Отображение графиков

# Сигналы
ax1.set_xlim(0, 0.18 * maxTime * dt)
ax1.set_ylim(-0.5, 1.2)
ax1.set_xlabel('t, с')
ax1.set_ylabel('Ez, В/м')
ax1.plot(tlist, Ez0)
ax1.plot(tlist, probe1Ez)
ax1.legend(['Падающий сигнал',
 'Отраженный сигнал'],
 loc='lower right')
ax1.minorticks_on()
ax1.grid()
Fmax = 2e9
Fmin = 0

# Спектры сигналов
ax2.set_xlim(Fmin, 1.25 * Fmax)
ax2.set_xlabel('f, Гц')
ax2.set_ylabel('|F{Ez}|, В*с/м')
ax2.plot(flist, Ez0Spec)
ax2.plot(flist, Ez1Spec)
ax2.legend(['Спектр падающего сигнала',
 'Спектр отраженного сигнала'],
 loc='upper right')
ax2.minorticks_on()
ax2.grid()

# Коэффициент отражения
ax3.set_xlim(Fmin, Fmax)
ax3.set_ylim(0, 1.0)
ax3.set_xlabel('f, Гц')
ax3.set_ylabel('|Г|, б/р')
ax3.plot(flist, Gamma)
ax3.minorticks_on()
ax3.grid()
plt.subplots_adjust(hspace=0.5)
plt.show()