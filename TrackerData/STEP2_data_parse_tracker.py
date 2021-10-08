#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:45:10 2021

@author: john
"""
import numpy as np
from csv import reader
from matplotlib import pyplot as plt
time = []
xvals = []
zvals = []
counter = 1
# open file in read mode
with open('selected_trajectories.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        if counter%2 == 1:
            xvals.append(row)
        else:
            zvals.append(row)
        counter+=1


# now build a 'time' variable knowing the nominal sampling rate is 125Hz
psample = 60
xfile = 'xdata.csv'
zfile = 'zdata.csv'
rfile = 'rdata.csv'
sampling_hz = 240
ms_per_sample = np.round(1/sampling_hz,3)
mts = []
jumps = []

all_x = np.zeros([np.size(zvals),70])
counter = 0
for trial in xvals:
    #extract the data from the trial_row
    infile = trial.pop(0)
    jump = np.int(trial.pop(0))
    jumps.append(jump)
    rt_sample = np.int(trial.pop(0))
    rt_ms = np.round(np.float64(trial.pop(0)),3)    
    mt_sample = np.int(trial.pop(0))
    mt_ms = np.round(np.float64(trial.pop(0)),3)
    mt = (mt_ms-rt_ms)
    mts.append(mt)
    which_axis = trial.pop(0)
    xdata = np.array(trial,dtype=float)
    xdata = xdata-xdata[psample-1]
    all_x[counter,:] = xdata[psample-3:psample+67]
    counter+=1
    print(counter)
                     

all_z = np.zeros([np.size(zvals),70])
counter = 0
for trial in zvals:
    #extract the data from the trial_row
    infile = trial.pop(0)
    jump = np.int(trial.pop(0))
    jumps.append(jump)
    rt_sample = np.int(trial.pop(0))
    rt_ms = np.round(np.float64(trial.pop(0)),3)    
    mt_sample = np.int(trial.pop(0))
    mt_ms = np.round(np.float64(trial.pop(0)),3)
    mt = (mt_ms-rt_ms)
    mts.append(mt)
    which_axis = trial.pop(0)
    ydata = np.array(trial,dtype=float)
    ydata = ydata-ydata[psample-1]
    all_z[counter,:] = ydata[psample-3:psample+67]
    counter+=1

all_res = np.zeros(np.shape(all_z))
rows,cols = np.shape(all_z)
for r in range(0,rows):
    for c in range(0,cols):
        all_res[r,c] = np.sqrt((all_x[r,c]**2) +(all_z[r,c]**2))

time = np.array([ms_per_sample*sample for sample in range(0,np.size(xdata))])
subtime = time[psample-3:psample+67]
subtime = subtime-0.250
mtime = [np.int(x*1000) for x in subtime]

(rs, cs) = np.shape(all_z)
for rows in range(0,rs):
    if jumps[rows] == 0:
        c = 'green'
    else:
        c = 'red'
    plt.plot(subtime,all_z[rows,:],color=c)
plt.plot([0,0],plt.ylim(),'blue')
plt.title('All-Y')
plt.show()
    
(rs, cs) = np.shape(all_x)
for rows in range(0,rs):
    if jumps[rows] == 0:
        c = 'green'
    else:
        c = 'red'
    plt.plot(subtime,all_x[rows,:],color=c)
plt.title('All-X')
plt.plot([0,0],plt.ylim(),'blue')
plt.show()

(rs, cs) = np.shape(all_x)
for rows in range(0,rs):
    if jumps[rows] == 0:
        c = 'green'
    else:
        c = 'red'
    plt.plot(all_x[rows,:],all_z[rows,:],color=c)   

plt.title('X-vs-Z Rescaled to Perturbation')
plt.show()

(rs, cs) = np.shape(all_x)
with open(xfile,'w') as xf:
    xf.write('Jump,')
    for i in range(0,np.size(mtime)):
        if i != np.size(mtime)-1:
            xf.write('t{}ms,'.format(mtime[i]))
        else:
            xf.write('t{}ms\n'.format(mtime[i]))
    for rows in range(0,rs):
        xf.write('{},'.format(jumps[rows]))
        for cols in range(0,cs):
            if cols != cs-1:
                xf.write('{:3f},'.format(all_x[rows,cols]))
            else:
                xf.write('{:3f}\n'.format(all_x[rows,cols]))
                
(rs, cs) = np.shape(all_z)              
with open(zfile,'w') as zf:
    zf.write('Jump,')
    for i in range(0,np.size(mtime)):
        if i != np.size(mtime)-1:
            zf.write('t{}ms,'.format(mtime[i]))
        else:
            zf.write('t{}ms\n'.format(mtime[i]))
    for rows in range(0,rs):
        zf.write('{},'.format(jumps[rows]))
        for cols in range(0,cs):
            if cols != cs-1:
                zf.write('{:3f},'.format(all_z[rows,cols]))
            else:
                zf.write('{:3f}\n'.format(all_z[rows,cols]))
                
(rs, cs) = np.shape(all_res)              
with open(rfile,'w') as rf:
    rf.write('Jump,')
    for i in range(0,np.size(mtime)):
        if i != np.size(mtime)-1:
            rf.write('t{}ms,'.format(mtime[i]))
        else:
            rf.write('t{}ms\n'.format(mtime[i]))
    for rows in range(0,rs):
        rf.write('{},'.format(jumps[rows]))
        for cols in range(0,cs):
            if cols != cs-1:
                rf.write('{:3f},'.format(all_res[rows,cols]))
            else:
                rf.write('{:3f}\n'.format(all_res[rows,cols]))            