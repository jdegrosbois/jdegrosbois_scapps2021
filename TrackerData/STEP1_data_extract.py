# -*- coding: utf-8 -*-
"""
Spyder Editor
## STILL NEED TO FITUGURE OUT AXES AND SCALING FROM THE PAPER... numbers to not make sense
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
goot_mts = []
good_trials = 0
files = glob.glob('*.mat')

x_max = []
y_max = []
z_max = []
x_min = []
y_min = []
z_min = []
x_first = []
y_first = []
z_first = []
x_last = []
y_last = []
z_last = []

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
        
    tracker_time = time[tracker_trials]
    tracker_xyz = xyz[tracker_trials]
    tracker_frame= frame[tracker_trials]
    tracker_delay = delay[tracker_trials]
    tracker_direction = direction[tracker_trials]
    tracker_block = block[tracker_trials]
    tracker_jump = jump[tracker_trials]
    
    rts = []
    mts = []
    trials = []
    
    for trial in range(0,190):#190):           
        if trial%25==0:
            print('......Now Processing trial: {}'.format(trial))
        
        ttime = np.squeeze(tracker_time[trial])
        txyz = tracker_xyz[trial]
        tframe = tracker_frame[trial]
        tdelay = tracker_delay[trial]
        tdirection = tracker_direction[trial]
        tblock = tracker_block[trial]
        tjump = tracker_jump[trial]
        
        
        # these trials have been removed by visual inspection
         # note that this is only for the jump trials
        if each_file == files[7] and trial == 147:
            continue # skip this trial       
 
        if each_file == files[9] and trial == 164:
            continue # skip this trial
            
        #if each_file == files[8] and trial == 77 and tdirection == 0:
        #    continue # skip this trial   
        
        #if each_file == files[6] and trial == 96 and tdirection == 0:
        #    continue # skip this trial   
 
        # now get x and y data for manipulating
        x = np.float64(txyz[0,:])
        y = np.float64(txyz[1,:])
        z = np.float64(txyz[2,:])
        
        # assuming tracker data is in mm rather than cm need to convert using mm/pixel
        #Remove thes and reinstate the above-three lines once you determine which axis the trackermovements hit the screen        
        x = ((x-x[0])*-1)*1/0.0341796875
        y = ((y-y[0])*1)*1/0.03671875
        z = ((z-z[0])*-1)*1/0.03671875 # cm per pixel Z assumng Z is actualy 'up', so same as Y in other conditions
                
        x_max.append(np.max(x))
        x_min.append(np.min(x))
        y_max.append(np.max(y))
        y_min.append(np.min(y))
        z_max.append(np.max(z))
        z_min.append(np.min(z))
        
        x_first.append(x[0])
        x_last.append(x[-1])
        y_first.append(y[0])
        y_last.append(y[-1])
        z_first.append(z[0])
        z_last.append(z[-1])
        
        Hz = 240#120 assumes downsampled as mentioned in the paper        
        # now adjust samping of x,y, and time to reflect the collection at 125 Hz
        #stamps = np.round(np.arange(ttime[0],ttime[-1],(1/Hz)),3) # actual sample times
        stamps = np.round(np.arange(0,len(x),1)*(1/Hz),3) # actual sample times
        rt = np.round(ttime,3)
        indices = []
        # now cycle through the stamps and find the first sample in rt that matches and save it        
        ttime = stamps
                
        #now filter with a lowpass 10Hz 2nd order butterworth
        #Hz = 120#240 assuming downsampled as mentioned in the paper
        filt_order = 2
        high_cut = 10
        x = butter_lowpass_filter(x,high_cut,Hz,filt_order)
        y = butter_lowpass_filter(y,high_cut,Hz,filt_order)
        z = butter_lowpass_filter(z,high_cut,Hz,filt_order)
        
        # use z to generate a rough RT estimate
        delta_t = ttime[1]-ttime[0]
        yvel = np.zeros(np.size(y))
        for ind, sample in enumerate(y):
            if ind > 0:
                yvel[ind] = (y[ind]-y[ind-1])/delta_t

        xvel = np.zeros(np.size(x))
        for ind, sample in enumerate(x):
            if ind > 0:
                xvel[ind] = (x[ind]-x[ind-1])/delta_t
        
        zvel = np.zeros(np.size(z))
        for ind, sample in enumerate(z):
            if ind > 0:
                zvel[ind] = (z[ind]-z[ind-1])/delta_t
        
        # compute resultant position and velocity
               
        res = np.sqrt(x**2+z**2)
        rvel = np.zeros(np.size(res))
        for ind, sample in enumerate(res):
            if ind > 0:
                rvel[ind] = (res[ind]-res[ind-1])/delta_t
        
        zyres = np.sqrt(x**2+z**2)
        zyrvel = np.zeros(np.size(zyres))
        for ind, sample in enumerate(zyres):
            if ind > 0:
                zyrvel[ind] = (zyres[ind]-zyres[ind-1])/delta_t
                
        RT = np.size(ttime)-3
        
        # check for RT with a 10mm/s threshold for 10 samples        
        for ind in range(0,np.size(zyrvel[0:-2])+1):
            if np.sum(zyrvel[ind:ind+2]>=10)==2:# changed zrvel crit to 10 from 50
                RT = ind
                rts.append(RT)
                #print(RT)
                break;
        
        MT = np.size(ttime)-2
        # check for movement end using a similar criteria in both x and z
        for ind in range(RT+10,np.size(rvel[0:-2])+1):
            if np.sum(np.abs(rvel[ind:ind+2])<=10)==2: # changed from 50 to 10 because all samples were <50
                MT = ind
                mts.append(MT)
                #print(RT)
                break;
                
        # find the position 100 ms after RT
        for ind, t in enumerate(ttime):
            if t > ttime[RT]+0.1:#0.2 changed to 0.1 for mvmt tracker
                early_sample = ind
                early_t = t
                break
                 
        if tdirection == 1:
            x = x*-1
        
        # if there is not enough data for an 'early sample' skip the trial
        if ttime[-1]- ttime[RT] < 0.300:
            continue
                
        # use trigonometry to find the angle between the vertical and this initial ballistica portion of the movement
        hyp = np.sqrt(x[early_sample]**2 +z[early_sample]**2)
        opp = z[early_sample]
        angle = np.rad2deg(np.arccos(opp/hyp))
        
        # based on description in manuscript, if you go straight at the side-target the angle would be 47 degree from the vertical
        half_angle = 47/2      
        
        #print("Trial {}; RT = {}; MT = {}; Direction = {}; Angle = {}".format(trial,ttime[RT],ttime[MT],tdirection,angle))
        #plt.plot(ttime,x)
        # if tdelay==0.25 and angle < (half_angle) and (ttime[RT] > 0.140 and ttime[RT] < 0.236):
        #     plt.plot(x,z)
        #     plt.ylim((-2,22))
        #     plt.xlim((-2,22))
        #     plt.xlabel('X')
        #     plt.ylabel('Z')
        #     plt.plot([x[0],x[early_sample]],[z[0],z[early_sample]],color='red')
        #     plt.plot([x[0],x[0]],[z[0],z[early_sample]],color='red')
        #     plt.title("Trial {}; Tdelay  = {}; RT = {}; MT = {}; Direction = {}; Angle = {}".format(trial,tdelay,ttime[RT],ttime[MT]-ttime[RT],tdirection,angle))
        #     plt.show()
        
        if tdelay==0.25 and (ttime[RT] > 0.140 and ttime[RT] < 0.236) and angle < (half_angle) and MT != 0 and ttime[MT]-ttime[RT] >= 0.3:
            #if True:
                
                
            #print('Mediant deltatime = {}'.format(1/(ttime[-1]/np.size(ttime))))
            #deltas.append(1/(ttime[-1]/np.size(ttime)))
            good_rts.append(ttime[RT])
            goot_mts.append(ttime[MT]-ttime[RT])
            trials.append(trial)
    
    
    
            #fix this to 3plot each trial            
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection = '3d')
    
            #ax.set_xlabel("X")
            #ax.set_ylabel("Z")
            #ax.set_zlabel("Y")
    
            #ax.scatter(x, z, y)
            #plt.show()
            
            
            # print('Telling it to plot...')
                        
            #plt.plot(x,y)
            if tdirection == 0:
                plt.plot(x, z,'green')
            else:
                plt.plot(x, z,'red')
            # plt.title(each_file)
            # plt.show()
            
            
            good_trials += 1
            # given that we have found a good trial (i.e., short enough Rt, small enough angle, and correct condition), write the x and y data to the outfile
            outfile.write('{},{},{},{},{},{},{},'.format(each_file,np.abs(tdirection),RT,ttime[RT],MT,ttime[MT],'x'))
            for value in x:
                if value == x[-1]:
                    outfile.write('{}'.format(value))
                else:
                    outfile.write('{},'.format(value))
            outfile.write('\n')
            outfile.write('{},{},{},{},{},{},{},'.format(each_file,np.abs(tdirection),RT,ttime[RT],MT,ttime[MT],'z'))
            for value in z:
                if value == z[-1]:
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
