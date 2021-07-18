import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections
import numpy as np
import argparse
import time

import tools.dec_tree_generator as tools
from scipy import signal
from scipy import fftpack

plt.style.use('fivethirtyeight')

last_pos = None
window_start_ts = None
window_end_ts = None
data_dir = "./data"
clf = None

parser = argparse.ArgumentParser(description='A Simple Visualization Tool')
parser.add_argument("--input_file_prefix", dest='file_prefix', help='input file prefix', default="chest")
parser.add_argument("--replay", dest='replay', help='replay as in the raw file', default=False, action='store_true')
parser.add_argument("--predict", dest='predict', help='predict', default=False, action='store_true')

args = parser.parse_args()
file_prefix = args.file_prefix
replay = args.replay
predict = args.predict

if replay :
    print("Replay raw input data...")

WL = 104

print("Visualize " + file_prefix + " data...")

def count_zero_crossing(queue, threshold) :

    count = 0

    for i in range(1, WL) :

        if queue[i-1] < threshold and threshold < queue[i] :
            count += 1
        if queue[i-1] > threshold and threshold > queue[i] :
            count += 1

    return count

def animate(i):
    
    #print("animate function...")
    global last_pos, window_end_ts, clf

    with open(data_dir + "/" + file_prefix + '_data.csv', 'r') as f : 
        #data = pd.read_csv('data.csv')

        # The fist time reading the input file
        if last_pos is None :

            if not replay :
                data = f.readlines()

                last_pos = f.tell()
                #print(last_pos)
                if len(data) == 0 :
                    print("Please check the input data, empty file...")
                    return
                else :
                    if len(data) ==  1: # The first line is header 
                        return
                    else :
                        data = data[1:]
            else :
                
                data = f.readline()
                last_pos = f.tell()
                return
        else :
            #print("=====================================")
            #print(last_pos)
            f.seek(last_pos)

            if not replay :
                data = f.readlines()

                last_pos = f.tell()
                #print(last_pos)
                if len(data) == 0 :
                    return

            else :
                data = []
                for _ in range(208) :

                    temp_data_line = f.readline()
                    if temp_data_line == '' :
                        return
                    last_pos = f.tell()
                    
                    data.append(temp_data_line)
                last_pos = f.tell()
                #print("line", data, type(data))

                if len(data) == 0 :
                    return
                

        ax_data_acc.cla()
        ax_data_gyr.cla()
        #print("line", data, type(data))

        for datum in data :

            #ch_ = datum.split(',')[0]        
            #print(ch_, type(ch_), ch_ == 'C')
            #print(datum)
            ts_ = int(datum.split(' ')[0])

            ax_ = int(datum.split(' ')[1])
            ay_ = int(datum.split(' ')[2])
            az_ = int(datum.split(' ')[3])

            gx_ = int(datum.split(' ')[4])
            gy_ = int(datum.split(' ')[5])
            gz_ = int(datum.split(' ')[6])

            prev_window_end_ts = data_ts.popleft()
            data_ts.append(ts_)
            
            data_ax.popleft()
            data_ax.append(ax_)

            data_ay.popleft()
            data_ay.append(ay_)

            data_az.popleft()
            data_az.append(az_)

            data_gx.popleft()
            data_gx.append(gx_)

            data_gy.popleft()
            data_gy.append(gy_)

            data_gz.popleft()
            data_gz.append(gz_)

            if window_end_ts == None :
                window_end_ts = ts_

            else :
                # FEATURE EXTRACTION
                if prev_window_end_ts == window_end_ts : 
                    window_end_ts = data_ts[-1]
                    #print(WL, " window filled...")

                    # Create data frame
                    series_names    = ["ts", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "acc_v", "acc_v2", "gyr_v", "gyr_v2"]
                    pd_series_ts    = pd.Series(list(data_ts),       range(1, WL+1), name=series_names[0])

                    pd_series_ax    = pd.Series(list(data_ax)[-WL:], range(1, WL+1), name=series_names[1]) / 1000.
                    pd_series_ay    = pd.Series(list(data_ay)[-WL:], range(1, WL+1), name=series_names[2]) / 1000.
                    pd_series_az    = pd.Series(list(data_az)[-WL:], range(1, WL+1), name=series_names[3]) / 1000.

                    pd_series_gx    = pd.Series(list(data_gx)[-WL:], range(1, WL+1), name=series_names[4]) / 1000. * 0.017453 # dps to rad/s
                    pd_series_gy    = pd.Series(list(data_gy)[-WL:], range(1, WL+1), name=series_names[5]) / 1000. * 0.017453
                    pd_series_gz    = pd.Series(list(data_gz)[-WL:], range(1, WL+1), name=series_names[6]) / 1000. * 0.017453

                    pd_series_acc_v     = (pd_series_ax.pow(2) + pd_series_ay.pow(2) + pd_series_az.pow(2)).pow(0.5)
                    pd_series_acc_v2    = pd_series_ax.pow(2) + pd_series_ay.pow(2) + pd_series_az.pow(2)

                    pd_series_gyr_v     = (pd_series_gx.pow(2) + pd_series_gy.pow(2) + pd_series_gz.pow(2)).pow(0.5)
                    pd_series_gyr_v2    = pd_series_gx.pow(2) + pd_series_gy.pow(2) + pd_series_gz.pow(2)

                    temp_dict = {
                        series_names[0]: pd_series_ts,
                        series_names[1]: pd_series_ax,
                        series_names[2]: pd_series_ay,
                        series_names[3]: pd_series_az,
                        series_names[4]: pd_series_gx,
                        series_names[5]: pd_series_gy,
                        series_names[6]: pd_series_gz,

                        series_names[7]: pd_series_acc_v,
                        series_names[8]: pd_series_acc_v2,
                        series_names[9]: pd_series_gyr_v,
                        series_names[10]: pd_series_gyr_v2,
                    }
                    pd_temp = pd.DataFrame(temp_dict)
                    #print(pd_temp) 
                    #print(pd_temp.mean(axis=0))
                    #print(pd_temp.max(axis=0)-pd_temp.min(axis=0))
                    #print("zero crossing of acc ax: ", count_zero_crossing(list(data_ax)[-104:], 0))
                    mean_pd_temp = pd_temp.mean(axis=0)
                    peak_to_peak_pd_temp = pd_temp.max(axis=0)-pd_temp.min(axis=0)
                    feature_list = [mean_pd_temp["acc_x"],         mean_pd_temp["acc_y"],         mean_pd_temp["acc_z"],         mean_pd_temp["acc_v"],         mean_pd_temp["acc_v2"],
                                    mean_pd_temp["gyr_x"],         mean_pd_temp["gyr_y"],         mean_pd_temp["gyr_z"],         mean_pd_temp["gyr_v"],         mean_pd_temp["gyr_v2"],
                                    peak_to_peak_pd_temp["acc_x"], peak_to_peak_pd_temp["acc_y"], peak_to_peak_pd_temp["acc_z"], peak_to_peak_pd_temp["acc_v"], peak_to_peak_pd_temp["acc_v2"], 
                                    peak_to_peak_pd_temp["gyr_x"], peak_to_peak_pd_temp["gyr_y"], peak_to_peak_pd_temp["gyr_z"], peak_to_peak_pd_temp["gyr_v"], peak_to_peak_pd_temp["gyr_v2"], 
                                    count_zero_crossing(list(data_ax)[-WL:], 0), count_zero_crossing(list(data_ax)[-WL:], 0), count_zero_crossing(list(data_ax)[-WL:], 0),
                                    count_zero_crossing(list(data_gx)[-WL:], 0), count_zero_crossing(list(data_gx)[-WL:], 0), count_zero_crossing(list(data_gx)[-WL:], 0)] 

                    #print("feature: ", feature_list) 

                    # Predict and show the results in the defined window size
                    if predict : 
                        y_pred = clf.predict([feature_list])
                        print("y_pred: ", y_pred, " probability: ", clf.predict_proba([feature_list]))

        # Filter test
        b, a = signal.butter(3, 0.02)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, data_ax, zi=zi*data_ay[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        z3, _ = signal.lfilter(b, a, z, zi=zi*z2[0])
        y = signal.filtfilt(b, a, data_ay)

        #X = fftpack.fft(y)
        #freqs = fftpack.fftfreq(len(y)) * 208 
        #print("freqs: ", freqs)

        ax_data_acc.plot(data_ax, linewidth=1, label="x")
        ax_data_acc.plot(y, 'r--', linewidth=1, label="y filtered")
        ax_data_acc.plot(data_ay, linewidth=1, label="y")
        pc = ax_data_acc.plot(data_az, linewidth=1, label="z")
        ax_data_acc.scatter(len(data_az)-1, data_az[-1], facecolor = pc[0].get_color())
        ax_data_acc.set_ylim(-4e3, 4e3)
        ax_data_acc.set_title(file_prefix + " accelerometer")
        ax_data_acc.legend(loc='upper left')

        ax_data_gyr.plot(data_gx, linewidth=1)
        ax_data_gyr.plot(data_gy, linewidth=1)
        pc = ax_data_gyr.plot(data_gz, linewidth=1)
        ax_data_gyr.scatter(len(data_gz)-1, data_gz[-1], facecolor = pc[0].get_color())
        ax_data_gyr.set_ylim(-4e6, 4e6)
        ax_data_gyr.set_title(file_prefix + " gyroscope")

# Window Length WL = 104
# Create a decision tree for stationary, stand up, sit down, walking
arff_filename    = "/Volumes/Samsung_T3/TomatoCrew/TomatoEmbedded/data/junsang/first_trial.arff"
dectree_filename = "/Volumes/Samsung_T3/TomatoCrew/TomatoEmbedded/data/junsang/dectree.txt"

clf = tools.generateDecisionTree(arff_filename, dectree_filename)

data_ts = collections.deque(np.zeros(WL)) 

data_ax = collections.deque(np.zeros(1000))
data_ay = collections.deque(np.zeros(1000))
data_az = collections.deque(np.zeros(1000))

data_gx = collections.deque(np.zeros(1000))
data_gy = collections.deque(np.zeros(1000))
data_gz = collections.deque(np.zeros(1000))

fig = plt.figure(figsize=(12,6), facecolor="#DEDEDE")

ax_data_acc = plt.subplot(121)
ax_data_gyr = plt.subplot(122)

ax_data_acc.set_title(file_prefix + " accelerometer")
ax_data_gyr.set_title(file_prefix + " gyroscope")

ax_data_acc.set_facecolor('#DEDEDE')
ax_data_gyr.set_facecolor('#DEDEDE')

ani = FuncAnimation(fig, animate, interval=1)
fig.tight_layout()
plt.show()
