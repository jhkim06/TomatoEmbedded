import serial
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import collections

import csv

#fieldnames = ["ch", "ts", "ax", "ay", "az", "gx", "gy", "gz"]
fieldnames = ["Timestamp[us]", "A_X[mg]", "A_Y[mg]", "A_Z[mg]", "G_X[mdps]", "G_Y[mdps]", "G_Z[mdps]"] # header format for UNICO GUI

data_dir = "./data"
out_file_name = 'data.csv'

with open(data_dir + "/" + "chest_" + out_file_name, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter = ' ')
    csv_writer.writeheader()

with open(data_dir + "/" + "wrist_" + out_file_name, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter = ' ')
    csv_writer.writeheader()

# Accelerometer: 208 Hz, 2g full scale, micro g
# Gyroscope: 208 Hz, 2000 dpgs full scale, micro dps

s = serial.Serial("/dev/tty.usbmodem0000000000001", 460800)

while s.isOpen() == False :
    pass

time.sleep(5) # CAUTION: THE ORDER OF THIS SLEEP REALLY MATTER, SLEEP BEFORE SEND START MESSAGE !!

print("Send START")
s.write(b'START')

try : 

    while True :

        # TODO Check if the ODR(Output Data Rate) for both sensors are 417 Hz.
    
        #bytesToRead = s.inWaiting()
        bytesToRead = 36
        packet = s.read(bytesToRead)
    
        # FIXME Check case with bytesToRead larget than 36 bytes (though, most of time it is 36 bytes)
        #if bytesToRead > 36 :
        #    print("bytesToRead: ", bytesToRead)
        #    for i in range(bytesToRead// 36) :
        #        start_byte = i * 36
        #        end_byte = start_byte + 36

        #        print(packet[start_byte:end_byte])
    
        try :
    
            #print(packet)
            channel_name = struct.unpack("c", packet[3:4])[0].decode()
            time_stamp = struct.unpack("I", packet[4:8])[0] #
            #print("channel, time_stamp: ", channel_name, time_stamp, packet[4:8])

            chest_aX, chest_aY, chest_aZ = struct.unpack("3i", packet[8:20])[0:3]
            chest_gX, chest_gY, chest_gZ = struct.unpack("3i", packet[20:32])[0:3]
    
            if channel_name == 'C' :
                file_prefix = "chest_"
            elif channel_name == 'W' :
                file_prefix = "wrist_"
            else :
                print("Cannot identifiy if it is chest or wrist data, skip...")
                print(packet)
                continue

            with open(data_dir + "/" + file_prefix + out_file_name, 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter = ' ')
    
                info = {
                    #"ch": channel_name,
                    #"ts": time_stamp,

                    fieldnames[0]: time_stamp , 

                    fieldnames[1]: chest_aX // 1000, # micro to milli
                    fieldnames[2]: chest_aY // 1000,
                    fieldnames[3]: chest_aZ // 1000,
                    fieldnames[4]: chest_gX // 1000,
                    fieldnames[5]: chest_gY // 1000,
                    fieldnames[6]: chest_gZ // 1000
                }
                csv_writer.writerow(info)
                #print(bytesToRead, struct.unpack("c", packet[3:4]), struct.unpack("6i", packet[8:32]))
    
        except BaseException as e :
            pass
    
except KeyboardInterrupt :

    print("")
    print("Stop data collecting...") 
    s.write(b'STOP')
    s.close()

