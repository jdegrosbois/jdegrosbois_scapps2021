# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.io import loadmat
from  matplotlib import pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
import glob

## Functions Section
################################################################################
def butter_lowpass(highcut, fs, order=2):
        nyq = 0.5 * fs
        high = highcut / nyq
        sos = butter(order, [high], analog=False, btype='lowpass', output='sos')
        return sos

def butter_lowpass_filter(data, highcut, fs, order=2):
        sos = butter_lowpass(highcut, fs, order=order)
        y = sosfiltfilt(sos, np.array(data),axis = 0) # need to use an array
        return y

# open an output file for the full-trajectories
outfile = open('selected_trajectories.csv','w')

good_rts = []
good_trials = 0
files = glob.glob('*.mat')

print('Starting crunch...')
for each_file in files:
    print('...Processing file: {}'.format(each_file))
    # load the matfile
    data = loadmat(each_file,squeeze_me=False,struct_as_record=True)
    # pull the main data-structure
    struct = data['result']
    
    #grab the individual nd-arrays
    xyz = struct['xyz'][0][0][0,:]
    time = struct['time'][0][0][0,:]
    xy = struct['xy'][0][0][0,:]
    frame = struct['frame'][0][0][0,:]
    device = struct['device'][0][0][:,0]
    delay = struct['delay'][0][0][:,0]
    direction = struct['direction'][0][0][:,0]
    block = struct['block'][0][0][:,0]
    jump = struct['jump'][0][0][:,0]
    
    # first break the data down into each device
    stylus_trials = device==2
    mouse_trials = device==1
    tracker_trials = device==0
        
    mouse_time = time[stylus_trials]
    mouse_xy = xy[stylus_trials]
    mouse_frame= frame[stylus_trials]
    mouse_delay = delay[stylus_trials]
    mouse_direction = direction[stylus_trials]
    mouse_block = block[stylus_trials]
    mouse_jump = jump[stylus_trials]
    
    rts = []
    mts = []
    trials = []
    
    for trial in range(0,190):           
        if trial%25==0:
            print('......Now Processing trial: {}'.format(trial))
        
        ttime = np.squeeze(mouse_time[trial])
        txy = mouse_xy[trial]
        tframe = mouse_frame[trial]
        tdelay = mouse_delay[trial]
        tdirection = mouse_direction[trial]
        tblock = mouse_block[trial]
        tjump = mouse_jump[trial]
        
        
        # these trials have been removed by visual inspection
         # note that this is only for the jump trials
        if each_file == files[1] and trial == 85 and tdirection != 0:
            continue # skip this trial       
 
        if each_file == files[8] and trial == 117 and tdirection != 0:
            continue # skip this trial
            
        if each_file == files[8] and trial == 77 and tdirection == 0:
            continue # skip this trial   
        
        if each_file == files[6] and trial == 96 and tdirection == 0:
            continue # skip this trial   
 
        # now get x and y data for manipulating
        x = np.float64(txy[0,:])
        y = np.float64(txy[1,:])
        
        # rescale x and y to mm and oriented upward from 0,0
        x = ((x - x[0])*-1)*0.341796875
        y = ((y-y[0])*-1)*0.3671875 # cm per pixel Y
        
        # now adjust samping of x,y, and time to reflect the collection at 125 Hz
        stamps = np.round(np.arange(ttime[0],ttime[-1],(1/125)),3) # actual sample times
        rt = np.round(ttime,3)
        indices = []
        # now cycle through the stamps and find the first sample in rt that matches and save it
        for i in stamps:
            for j, k in enumerate(rt):
                if k==i:
                    indices.append(j)
                    break
        
        ttime = ttime[indices]
        x = x[indices]
        y = y[indices]
                
        #now filter with a lowpass 10Hz 2nd order butterworth
        Hz = 125
        filt_order = 2
        high_cut = 10
        x = butter_lowpass_filter(x,high_cut,Hz,filt_order)
        y = butter_lowpass_filter(y,high_cut,Hz,filt_order)
        
        # use y to generate a rough RT estimate
        delta_t = ttime[1]-ttime[0]
        yvel = np.zeros(np.size(y))
        for ind, sample in enumerate(y):
            if ind > 0:
                yvel[ind] = (y[ind]-y[ind-1])/delta_t

        xvel = np.zeros(np.size(x))
        for ind, sample in enumerate(x):
            if ind > 0:
                xvel[ind] = (x[ind]-x[ind-1])/delta_t
        
        # compute resultant position and velocity
        res = np.sqrt(x**2+y**2)
        rvel = np.zeros(np.size(res))
        for ind, sample in enumerate(res):
            if ind > 0:
                rvel[ind] = (res[ind]-res[ind-1])/delta_t
        
        # check for RT with a 10mm/s threshold for 10 samples        
        for ind in range(0,np.size(yvel[0:-2])+1):
            if np.sum(yvel[ind:ind+2]>=50)==2:
                RT = ind
                rts.append(RT)
                #print(RT)
                break;
        
        MT = 0
        # check for movement end using a similar criteria in both x and y
        for ind in range(RT+10,np.size(rvel[0:-2])+1):
            if np.sum(np.abs(rvel[ind:ind+2])<=50)==2:
                MT = ind
                mts.append(MT)
                #print(RT)
                break;
                
        # find the position 100 ms after RT
        for ind, t in enumerate(ttime):
            if t > ttime[RT]+0.2:
                early_sample = ind
                early_t = t
                break
                 
        if tdirection == 1:
            x = x*-1
        
        # use trigonometry to find the angle between the vertical and this initial ballistica portion of the movement
        hyp = np.sqrt(x[early_sample]**2 +y[early_sample]**2)
        opp = y[early_sample]
        angle = np.rad2deg(np.arccos(opp/hyp))
        
        # based on description in manuscript, if you go straight at the side-target the angle would be 47 degree from the vertical
        half_angle = 47/2      
        
        if tdelay==0.25 and (ttime[RT] > 0.140 and ttime[RT] < 0.225) and angle < (half_angle) and MT != 0 and ttime[MT]-ttime[RT] >= 0.3:
            #print('Mediant deltatime = {}'.format(1/(ttime[-1]/np.size(ttime))))
            #deltas.append(1/(ttime[-1]/np.size(ttime)))
            good_rts.append(ttime[RT])
            trials.append(trial)
            #plt.plot(x,y)
            if tdirection == 0:
                plt.plot(x,y,color='green')
            else:
                plt.plot(x,y,color='red')
            plt.title(each_file)
            good_trials += 1
            # given that we have found a good trial (i.e., short enough Rt, small enough angle, and correct condition), write the x and y data to the outfile
            outfile.write('{},{},{},{},{},{},{},'.format(each_file,np.abs(tdirection),RT,ttime[RT],MT,ttime[MT],'x'))
            for value in x:
                if value == x[-1]:
                    outfile.write('{}'.format(value))
                else:
                    outfile.write('{},'.format(value))
            outfile.write('\n')
            outfile.write('{},{},{},{},{},{},{},'.format(each_file,np.abs(tdirection),RT,ttime[RT],MT,ttime[MT],'y'))
            for value in y:
                if value == y[-1]:
                    outfile.write('{}'.format(value))
                else:
                    outfile.write('{},'.format(value))
            outfile.write('\n')
    #plt.title('File: {}'.format(each_file))
    #plt.xlim(-50,200)
    #plt.show()

plt.title('All Saved Trials')
plt.xlim(-50,200)
plt.show()
print('All Good Trials = {}'.format(good_trials))
print('All Good RTs:')
print(good_rts)
outfile.flush()
outfile.close()