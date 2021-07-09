import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections
import numpy as np

plt.style.use('fivethirtyeight')

last_pos = None
data_dir = "./data"

def animate(i):
    
    #print("animate function...")
    global last_pos

    with open(data_dir + "/" + 'chest.csv', 'r') as f : 
        #data = pd.read_csv('chest.csv')

        if last_pos is None :
            data = f.readlines()
            last_pos = f.tell()
            #print(last_pos)
            if len(data) == 0 :
                return
            else :
                data = data[1:]
                if data[0] == 'ch,ts,ax,ay,az,gx,gy,gz' :
                    return
        else :
            #print("=====================================")
            #print(last_pos)
            f.seek(last_pos)
            data = f.readlines()
            last_pos = f.tell()
            #print(last_pos)
            if len(data) == 0 :
                return

        ax_chest_acc.cla()
        ax_chest_gyr.cla()
        ax_wrist_acc.cla()
        ax_wrist_gyr.cla()
        #print("line", data, type(data))

        for datum in data :

            ch_ = datum.split(',')[0]        
            #print(ch_, type(ch_), ch_ == 'C')

            ax_ = int(datum.split(',')[2])
            ay_ = int(datum.split(',')[3])
            az_ = int(datum.split(',')[4])

            gx_ = int(datum.split(',')[5])
            gy_ = int(datum.split(',')[6])
            gz_ = int(datum.split(',')[7])
            #print(index, data)
            #print(az)
            #print(chest)

            
            if ch_ == 'C' : 
                chest_ax.popleft()
                chest_ax.append(ax_)

                chest_ay.popleft()
                chest_ay.append(ay_)

                chest_az.popleft()
                chest_az.append(az_)

                chest_gx.popleft()
                chest_gx.append(gx_)

                chest_gy.popleft()
                chest_gy.append(gy_)

                chest_gz.popleft()
                chest_gz.append(gz_)

            if ch_ == "W" :
                wrist_ax.popleft()
                wrist_ax.append(ax_)

                wrist_ay.popleft()
                wrist_ay.append(ay_)

                wrist_az.popleft()
                wrist_az.append(az_)

                wrist_gx.popleft()
                wrist_gx.append(gx_)

                wrist_gy.popleft()
                wrist_gy.append(gy_)

                wrist_gz.popleft()
                wrist_gz.append(gz_)

        ax_chest_acc.plot(chest_ax, linewidth=1, label="x")
        ax_chest_acc.plot(chest_ay, linewidth=1, label="y")
        pc = ax_chest_acc.plot(chest_az, linewidth=1, label="z")
        ax_chest_acc.scatter(len(chest_az)-1, chest_az[-1], facecolor = pc[0].get_color())
        ax_chest_acc.set_ylim(-5e6, 5e6)
        ax_chest_acc.set_title("Chest accelerometer")
        ax_chest_acc.legend(loc='upper left')

        ax_chest_gyr.plot(chest_gx, linewidth=1)
        ax_chest_gyr.plot(chest_gy, linewidth=1)
        pc = ax_chest_gyr.plot(chest_gz, linewidth=1)
        ax_chest_gyr.scatter(len(chest_gz)-1, chest_gz[-1], facecolor = pc[0].get_color())
        ax_chest_gyr.set_ylim(-5e8, 5e8)
        ax_chest_gyr.set_title("Chest gyroscope")

        ax_wrist_acc.plot(wrist_ax, linewidth=1)
        ax_wrist_acc.plot(wrist_ay, linewidth=1)
        pc = ax_wrist_acc.plot(wrist_az, linewidth=1)
        ax_wrist_acc.scatter(len(wrist_az)-1, wrist_az[-1], facecolor = pc[0].get_color())
        ax_wrist_acc.set_ylim(-5e6, 5e6)
        ax_wrist_acc.set_title("Wrist accelerometer")

        ax_wrist_gyr.plot(wrist_gx, linewidth=1)
        ax_wrist_gyr.plot(wrist_gy, linewidth=1)
        pc = ax_wrist_gyr.plot(wrist_gz, linewidth=1)
        ax_wrist_gyr.scatter(len(wrist_gz)-1, wrist_gz[-1], facecolor = pc[0].get_color())
        ax_wrist_gyr.set_ylim(-5e8, 5e8)
        ax_wrist_gyr.set_title("Wrist gyroscope")

# Chest sensor
chest_ax = collections.deque(np.zeros(1000))
chest_ay = collections.deque(np.zeros(1000))
chest_az = collections.deque(np.zeros(1000))

chest_gx = collections.deque(np.zeros(1000))
chest_gy = collections.deque(np.zeros(1000))
chest_gz = collections.deque(np.zeros(1000))

# Wrist sensor
wrist_ax = collections.deque(np.zeros(1000))
wrist_ay = collections.deque(np.zeros(1000))
wrist_az = collections.deque(np.zeros(1000))

wrist_gx = collections.deque(np.zeros(1000))
wrist_gy = collections.deque(np.zeros(1000))
wrist_gz = collections.deque(np.zeros(1000))

fig = plt.figure(figsize=(12,6), facecolor="#DEDEDE")

ax_chest_acc = plt.subplot(221)
ax_chest_gyr = plt.subplot(222)
ax_wrist_acc = plt.subplot(223)
ax_wrist_gyr = plt.subplot(224)

ax_chest_acc.set_title("Chest accelerometer")
ax_chest_gyr.set_title("Chest gyroscope")
ax_wrist_acc.set_title("Wrist accelerometer")
ax_wrist_gyr.set_title("Wrist gyroscope")

ax_chest_acc.set_facecolor('#DEDEDE')
ax_chest_gyr.set_facecolor('#DEDEDE')
ax_wrist_acc.set_facecolor('#DEDEDE')
ax_wrist_gyr.set_facecolor('#DEDEDE')

ani = FuncAnimation(fig, animate, interval=1)
fig.tight_layout()
plt.show()
