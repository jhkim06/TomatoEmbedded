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
import csv

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
parser.add_argument("--make_attr_file", dest='make_attr_file', help='make a attribute file', default=False, action='store_true')

args = parser.parse_args()
file_prefix = args.file_prefix
replay = args.replay
predict = args.predict
make_attr_file = args.make_attr_file

if predict :
    model = tf.keras.models.load_model('/Volumes/Samsung_T3/TomatoCrew/TomatoEmbedded/tools/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN/test_model')

if replay :
    print("Replay raw input data...")
fs = 208 # 208, 50
WL = fs * 3
data_window = 1000

attribute_list_for_dt = ['acc_v_mean', 'gyr_v_mean', 'acc_v_peak_to_peak', 'gyr_v_peak_to_peak', 'hz', 'label']

if make_attr_file :

    with open(data_dir + "/" + file_prefix + '_data.arff', 'w') as out_arff_file:
        csv_writer = csv.writer(out_arff_file) 
        csv_writer.writerow(["@data"])


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
                for _ in range(fs) :

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

            data_av.popleft()
            av_ = pow(pow(ax_, 2) + pow(ay_, 2) + pow(az_, 2), 0.5)
            data_av.append(av_)

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

                    #pd_series_ax    = pd.Series(list(data_ax)[-WL:], range(1, WL+1), name=series_names[1]) / 1000.
                    #pd_series_ay    = pd.Series(list(data_ay)[-WL:], range(1, WL+1), name=series_names[2]) / 1000.
                    #pd_series_az    = pd.Series(list(data_az)[-WL:], range(1, WL+1), name=series_names[3]) / 1000.

                    #pd_series_gx    = pd.Series(list(data_gx)[-WL:], range(1, WL+1), name=series_names[4]) / 1000. * 0.017453 # dps to rad/s
                    #pd_series_gy    = pd.Series(list(data_gy)[-WL:], range(1, WL+1), name=series_names[5]) / 1000. * 0.017453
                    #pd_series_gz    = pd.Series(list(data_gz)[-WL:], range(1, WL+1), name=series_names[6]) / 1000. * 0.017453

                    pd_series_ax    = pd.Series(list(data_ax)[-WL:], range(1, WL+1), name=series_names[1]) 
                    pd_series_ay    = pd.Series(list(data_ay)[-WL:], range(1, WL+1), name=series_names[2]) 
                    pd_series_az    = pd.Series(list(data_az)[-WL:], range(1, WL+1), name=series_names[3]) 

                    pd_series_gx    = pd.Series(list(data_gx)[-WL:], range(1, WL+1), name=series_names[4]) 
                    pd_series_gy    = pd.Series(list(data_gy)[-WL:], range(1, WL+1), name=series_names[5]) 
                    pd_series_gz    = pd.Series(list(data_gz)[-WL:], range(1, WL+1), name=series_names[6]) 


                    pd_series_acc_v     = (pd_series_ax.pow(2) + pd_series_ay.pow(2) + pd_series_az.pow(2)).pow(0.5)
                    pd_series_acc_v2    = pd_series_ax.pow(2) + pd_series_ay.pow(2) + pd_series_az.pow(2)

                    pd_series_gyr_v     = (pd_series_gx.pow(2) + pd_series_gy.pow(2) + pd_series_gz.pow(2)).pow(0.5)
                    pd_series_gyr_v2    = pd_series_gx.pow(2) + pd_series_gy.pow(2) + pd_series_gz.pow(2)

                    # Filter test
                    #b, a = signal.butter(3, 0.1) # fs = 50 Hz
                    b, a = signal.butter(3, 0.02) # fs = 208 Hz
                    zi = signal.lfilter_zi(b, a)
                    z, _ = signal.lfilter(b, a, pd_series_acc_v.to_numpy(), zi=zi*pd_series_acc_v.to_numpy()[0])
                    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
                    z3, _ = signal.lfilter(b, a, z2, zi=zi*z2[0])
                    y = signal.filtfilt(b, a, pd_series_acc_v.to_numpy())

                    pd_series_filtered_acc_v = pd.Series(y, range(1, WL+1), name="filtered_acc_v")

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
                        "filtered_acc_v": pd_series_filtered_acc_v
                    }                    


                    pd_temp = pd.DataFrame(temp_dict)

                    X = pd_temp[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']]
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    scaled_X = pd.DataFrame(data = X, columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])

                    cnn_input = [[scaled_X['acc_x'].values[:], scaled_X['acc_x'].values[:], scaled_X['acc_x'].values[:], scaled_X['acc_x'].values[:], scaled_X['acc_x'].values[:], scaled_X['acc_x'].values[:]]]
                    cnn_input = np.asarray(cnn_input).reshape(-1, WL, 6, 1)
                    #cnn_input.reshape(1, 200, 6, 1)
                    #(1, 200, 6)
                    #print(cnn_input.shape)
                    
                    #print(pd_temp) 
                    #print(pd_temp.mean(axis=0))
                    #print(pd_temp.max(axis=0)-pd_temp.min(axis=0))
                    #print("zero crossing of acc ax: ", count_zero_crossing(list(data_ax)[-104:], 0))
                    try :
                        mean_pd_temp = pd_temp.mean(axis=0)
                    except :
                        print(pd_temp)
                    peak_to_peak_pd_temp = pd_temp.max(axis=0)-pd_temp.min(axis=0)
                    #feature_list = [mean_pd_temp["acc_x"],         mean_pd_temp["acc_y"],         mean_pd_temp["acc_z"],         mean_pd_temp["acc_v"],         mean_pd_temp["acc_v2"],
                    #                mean_pd_temp["gyr_x"],         mean_pd_temp["gyr_y"],         mean_pd_temp["gyr_z"],         mean_pd_temp["gyr_v"],         mean_pd_temp["gyr_v2"],
                    #                peak_to_peak_pd_temp["acc_x"], peak_to_peak_pd_temp["acc_y"], peak_to_peak_pd_temp["acc_z"], peak_to_peak_pd_temp["acc_v"], peak_to_peak_pd_temp["acc_v2"], 
                    #                peak_to_peak_pd_temp["gyr_x"], peak_to_peak_pd_temp["gyr_y"], peak_to_peak_pd_temp["gyr_z"], peak_to_peak_pd_temp["gyr_v"], peak_to_peak_pd_temp["gyr_v2"], 
                    #                count_zero_crossing(list(data_ax)[-WL:], 0), count_zero_crossing(list(data_ax)[-WL:], 0), count_zero_crossing(list(data_ax)[-WL:], 0),
                    #                count_zero_crossing(list(data_gx)[-WL:], 0), count_zero_crossing(list(data_gx)[-WL:], 0), count_zero_crossing(list(data_gx)[-WL:], 0)] 



                    peak_positions = signal.find_peaks(y, distance = fs//2)[0]
                    peaks = [y[position] for position in peak_positions]

                    np_fft = np.fft.fft(y)
                    amplitudes = 2 / WL * np.abs(np_fft)
                    frequencies = np.fft.fftfreq(WL) * WL * 1 / (WL * 1./fs)
                    #print("amplitude: ", amplitudes[1:WL//2].max(), amplitudes[1:WL//2].argmax()) # amplitude and frequency
                    #print("freq: ", frequencies[1:WL//2][amplitudes[1:WL//2].argmax()])
                    #print("amplitude: ", amplitudes[:WL//2][:10])

                    #print(np.mean(peaks)-mean_pd_temp["filtered_acc_v"])
                    #print("zero count upper: ", count_zero_crossing(y, 50. + mean_pd_temp["filtered_acc_v"] ))
                    #print("zero count lower: ", count_zero_crossing(y, mean_pd_temp["filtered_acc_v"] - 50. ))
                    #print("final feature: ", min(count_zero_crossing(y, 50. + mean_pd_temp["filtered_acc_v"]), count_zero_crossing(y, mean_pd_temp["filtered_acc_v"] - 50. ))// len(peaks))

                    try :
                        filtered_crossing = min(count_zero_crossing(y, 50. + mean_pd_temp["filtered_acc_v"]), count_zero_crossing(y, mean_pd_temp["filtered_acc_v"] - 50. ))/ len(peaks)
                    except :
                        filtered_crossing = 0

                    ax_data_acc.plot(pd_series_acc_v.to_numpy(), linewidth=1, label="v")
                    ax_data_acc.plot(y, 'r--', linewidth=1, label="v filtered")
                    ax_data_acc.scatter(peak_positions, peaks, facecolor='black')
                    ax_data_acc.plot(range(1, WL+1), [mean_pd_temp["filtered_acc_v"]]*WL, linewidth=1, label="mean")

                    # Find peak to peak
                    peak_diff_list = np.diff(peak_positions)

                    if len(peak_diff_list) == 0 :
                        filtered_hz = 0.
                    else :
                        filtered_hz = 1. / (np.mean(peak_diff_list) * 1./ fs)

                    #print(signal.find_peaks(y)[0])
                    #
                    #try :
                    #    hz = 1. / (np.mean(np.diff(signal.find_peaks(y)[0])) * 1./ fs)
                    #    print("mean Hz: ", hz)
                    #except :
                    #    print(np.diff(signal.find_peaks(y)[0]))
                    #    hz = 0

                    #if np.nan == hz :
                    #    hz = 0.
                    #if np.inf == hz :
                    #    hz = 0.

                    feature_list = [mean_pd_temp["acc_v"], mean_pd_temp["gyr_v"], peak_to_peak_pd_temp["acc_v"], peak_to_peak_pd_temp["gyr_v"], filtered_crossing, filtered_hz]

                    if make_attr_file :
                    
                        with open(data_dir + "/" + file_prefix + '_data.arff', 'a') as out_arff_file:
                            csv_writer = csv.writer(out_arff_file) 

                            # attribute_list_for_dt = ['acc_v_mean', 'gyr_v_mean', 'acc_v_peak_to_peak', 'gyr_v_peak_to_peak', 'hz', 'label']
                            csv_writer.writerow( feature_list + [file_prefix.split("_")[1]])
                            # filtered peak_to_peak
 

                    print("feature: ", feature_list) 

                    # Predict and show the results in the defined window size
                    if predict : 
                        try :
                            y_pred = clf.predict([feature_list])
                            print("y_pred: ", y_pred, " probability: ", clf.predict_proba([feature_list]))
                            ax_data_acc.text(0.5, 0.5, y_pred)
                        except :
                            print(feature_list)

                        #print(model.predict(cnn_input))
                        print(model.predict_classes(cnn_input))

        ## Filter test
        #b, a = signal.butter(3, 0.15) # 0.01
        #zi = signal.lfilter_zi(b, a)
        #z, _ = signal.lfilter(b, a, data_av, zi=zi*data_av[0])
        #z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        #z3, _ = signal.lfilter(b, a, z, zi=zi*z2[0])
        #y = signal.filtfilt(b, a, data_av)

        #np_fft = np.fft.fft(y)
        #amplitudes = 2 / 200 * np.abs(np_fft) 
        #frequencies = np.fft.fftfreq(200) * 200 * 1 / (200 * 1./50)
        ##print("amplitude: ", amplitudes[1:100].max(), amplitudes[1:100].argmax()) # amplitude and frequency
        ##print("freq: ", frequencies)

        ## Find peak to peak

        ##print(signal.find_peaks(y)[0])
        #peak_positions = signal.find_peaks(y, distance = 20)[0]
        #peaks = [y[position] for position in peak_positions]
        #print(peak_positions)
        #print("mean Hz: ", 1. / (np.mean(np.diff(signal.find_peaks(y)[0])) * 1./ fs) ) 

        #sos = signal.butter(3, 3., 'lp', fs = 208, output='sos')
        #y = signal.sosfilt(sos, data_ay)

        #X = fftpack.fft(y)
        #freqs = fftpack.fftfreq(len(y)) * 208 
        #print("freqs: ", freqs)

        ax_data_acc.plot(data_ax, linewidth=1, label="x")
        #ax_data_acc.plot(y, 'r--', linewidth=1, label="v filtered")
        ax_data_acc.plot(data_ay, linewidth=1, label="y")
        pc = ax_data_acc.plot(data_az, linewidth=1, label="z")
        #ax_data_acc.plot(data_av, linewidth=1, label="v")
        ax_data_acc.scatter(len(data_az)-1, data_az[-1], facecolor = pc[0].get_color())
        #ax_data_acc.scatter(peak_positions, peaks, facecolor='black')
        ax_data_acc.set_ylim(-4e3, 4e3)
        ax_data_acc.set_title(file_prefix + " accelerometer")
        ax_data_acc.legend(loc='upper left')

        ax_data_gyr.plot(data_gx, linewidth=1)
        ax_data_gyr.plot(data_gy, linewidth=1)
        pc = ax_data_gyr.plot(data_gz, linewidth=1)
        ax_data_gyr.scatter(len(data_gz)-1, data_gz[-1], facecolor = pc[0].get_color())
        ax_data_gyr.set_ylim(-4e5, 4e5)
        ax_data_gyr.set_title(file_prefix + " gyroscope")

# Window Length WL = 104
# Create a decision tree for stationary, stand up, sit down, walking
arff_filename    = "/Volumes/Samsung_T3/TomatoCrew/TomatoEmbedded/data/test_v2.arff"
dectree_filename = "/Volumes/Samsung_T3/TomatoCrew/TomatoEmbedded/data/test_dectree.txt"

clf = tools.generateDecisionTree(arff_filename, dectree_filename)

data_ts = collections.deque(np.zeros(WL)) 


data_ax = collections.deque(np.zeros(data_window))
data_ay = collections.deque(np.zeros(data_window))
data_az = collections.deque(np.zeros(data_window))
data_av = collections.deque(np.zeros(data_window))

data_gx = collections.deque(np.zeros(data_window))
data_gy = collections.deque(np.zeros(data_window))
data_gz = collections.deque(np.zeros(data_window))

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
