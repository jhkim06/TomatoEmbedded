import serial
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import collections

import csv

#fieldnames = ["ch", "ts", "ax", "ay", "az", "gx", "gy", "gz"]
#fieldnames = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
fieldnames = ["A_X[mg]", "A_Y[mg]", "A_Z[mg]", "G_X[dps]", "G_Y[dps]", "G_Z[dps]"]

data_dir = "./data"
out_file_name = 'data.csv'

with open(data_dir + "/" + "chest_" + out_file_name, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter = ' ')
    csv_writer.writeheader()

with open(data_dir + "/" + "wrist_" + out_file_name, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter = ' ')
    csv_writer.writeheader()

s = serial.Serial("/dev/tty.usbmodem0000000000001", 460800)

while s.isOpen() == False :
    pass

time.sleep(5) # CAUTION: THE ORDER OF THIS SLEEP REALLY MATTER, SLEEP BEFORE SEND START MESSAGE !!

print("Send START")
s.write(b'START')

try : 

    while True :
    
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
    
            channel_name = struct.unpack("c", packet[3:4])[0].decode()
            time_stamp = struct.unpack("i", packet[4:8])[0]

            chest_aX = struct.unpack("6i", packet[8:32])[0]
            chest_aY = struct.unpack("6i", packet[8:32])[1]
            chest_aZ = struct.unpack("6i", packet[8:32])[2]
    
            chest_gX = struct.unpack("6i", packet[8:32])[3]
            chest_gY = struct.unpack("6i", packet[8:32])[4]
            chest_gZ = struct.unpack("6i", packet[8:32])[5]

            #print(packet)
    
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

                    fieldnames[0]: chest_aX,
                    fieldnames[1]: chest_aY,
                    fieldnames[2]: chest_aZ,
                    fieldnames[3]: chest_gX,
                    fieldnames[4]: chest_gY,
                    fieldnames[5]: chest_gZ
                }
                csv_writer.writerow(info)
                #print(bytesToRead, struct.unpack("c", packet[3:4]), struct.unpack("6i", packet[8:32]))
    
            #print(bytesToRead, struct.unpack("c", packet[3:4]), chest_aX, chest_aY, chest_aZ)
            #print(bytesToRead, struct.unpack("c", packet[3:4]), struct.unpack("3i", packet[8:20]))
    
            #if bytesToRead > 36 :
            #    print(bytesToRead, struct.unpack("c", packet[3:4]), chest_aX, chest_aY, chest_aZ)
    
            #    chest_aX = struct.unpack("3i", packet[8+36:20+36])[0]
            #    chest_aY = struct.unpack("3i", packet[8+36:20+36])[1]
            #    chest_aZ = struct.unpack("3i", packet[8+36:20+36])[2]
            #    print(bytesToRead, struct.unpack("c", packet[39:4+36]), chest_aX, chest_aY, chest_aZ)
    
        except BaseException as e :
            pass
    
except KeyboardInterrupt :

    print("")
    print("Stop data collecting...") 
    s.write(b'STOP')
    s.close()

