#https://stackoverflow.com/questions/36700404/tensorflow-opening-log-data-written-by-summarywriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt




event_acc = EventAccumulator('logs/pose_rwth_lstm/version_0/events.out.tfevents.1672964012.LAPTOP-JV3O1OGP.27744.0')
#event_acc = EventAccumulator('logs/pose_rwth_st_gcn/version_0/events.out.tfevents.1673178968.LAPTOP-JV3O1OGP.24876.0')
#event_acc = EventAccumulator('logs/pose_rwth_bert/version_10/events.out.tfevents.1673196596.LAPTOP-JV3O1OGP.33148.0')
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

train_loss = event_acc.Scalars('train_loss')
train_acc = event_acc.Scalars('train_acc')

train_loss = [(s.step, s.value) for s in event_acc.Scalars('train_loss')]
train_acc= [(s.step, s.value) for s in event_acc.Scalars('train_acc')]
#convertable into dataframe

steps = [ t[0] for t in train_loss]
cross_entropy =  [t[1] for t in train_loss]



#import matplotlib.pyplot as plt

#fig, ax  = plt.subplots()

#ax.plot(steps, cross_entropy, linewidth= 0.5, color = 'k')
#ax.set_xlabel('epochs')
#ax.set_ylabel('crossentropy')


#plt.show()



#plt.rc('font', family='serif')
#plt.rc('xtick', labelsize='x-small')
#plt.rc('ytick', labelsize='x-small')

#fig = plt.figure(figsize=(4, 3))
#ax = fig.add_subplot(1, 1,1)

#ax.plot(steps, cross_entropy, color='k', ls='-')

#ax.set_xlabel('epochs')
#ax.set_ylabel('Crossentropy')
#plt.show()







#Smoothing the line
#https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
#from scipy.interpolate import make_interp_spline, BSpline
#import numpy as np

#xnew = np.linspace(min(steps), max(steps), 300)
#spl = make_interp_spline(steps, cross_entropy, k=3)
#cross_entropy_smooth = spl(xnew)


#plt.plot(xnew, cross_entropy_smooth)
#plt.show()



#from scipy.ndimage.filters import gaussian_filter1d

#ysmoothed = gaussian_filter1d(cross_entropy, sigma=5)
#plt.plot(steps, ysmoothed)
#plt.show()



#one of the easiest implementations I found was to use that Exponential Moving Average the
#Tensorboard uses:


def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    print('last:', last)
    smoothed = list()
    for point in scalars:
        #smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        #smoothed_val = last * weight + (1 - weight) * point
        smoothed_val = last * weight + (1 - weight) * last
        print(point)
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
        print("smoothed_val:", smoothed_val)
    return smoothed




weight = int(0.9)
plt.plot( steps, cross_entropy)
plt.show()







#https://stackoverflow.com/questions/60683901/tensorboard-smoothing

import matplotlib.pyplot as plt
import pandas as pd

df_train_loss = pd.DataFrame(train_loss, columns = ['epochs', 'cross_entropy'])


df_train_acc = pd.DataFrame(train_acc, columns = ['epochs', 'cross_entropy'])
#s = df.iloc[:, 0]

#plt.plot(df.iloc[:, 0], df.iloc[:, 1])


TSBOARD_SMOOTHING = [0.5, 0.85, 0.99]



df = df_train_acc
#for train accuracy
smooth = []
for ts_factor in TSBOARD_SMOOTHING:
    smooth.append(df_train_acc.ewm(alpha=(1 - ts_factor)).mean())

#accuracy

for ptx in range(3):
    #print(ptx)
    ptx = 2
    #font settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    #how many plots need to be assembled
    #plt.subplot(1,3,ptx+1)
    #plt.figure(figsize=(3.8, 2.8))
    plt.plot(df["cross_entropy"], alpha=0.4, color = '0.5', linewidth = 0.5)
    plt.plot(smooth[ptx]["cross_entropy"],color = 'k', linewidth= 0.5 )
    #plt.title("Tensorboard Smoothing = {}".format(TSBOARD_SMOOTHING[ptx]))
    plt.xlabel('epochs', fontsize = 9)
    #plt.ylabel('train_loss')
    plt.ylabel('train_acc')
    #plt.grid(alpha=0.3)
plt.savefig('X:/HandTalk/Handtalk/Ausarbeitung/Abbildungen/foo_train_acc_lstm_epochs.pdf')


#loss
plt.show()


df = df_train_loss

#for train loss
smooth = []
for ts_factor in TSBOARD_SMOOTHING:
    smooth.append(df.ewm(alpha=(1 - ts_factor)).mean())


for ptx in range(3):
    #print(ptx)
    ptx = 2
    #font settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    #how many plots need to be assembled
    #plt.subplot(1,3,ptx+1)
    #plt.figure(figsize=(3.8, 2.8))
    plt.plot(df["cross_entropy"], alpha=0.4, color = '0.5', linewidth = 0.5)
    plt.plot(smooth[ptx]["cross_entropy"],color = 'k', linewidth= 0.5 )
    #plt.title("Tensorboard Smoothing = {}".format(TSBOARD_SMOOTHING[ptx]))
    plt.xlabel('epochs', fontsize = 9)
    plt.ylabel('train_loss')
    #plt.ylabel('train_acc')
    #plt.grid(alpha=0.3)
plt.savefig('X:/HandTalk/Handtalk/Ausarbeitung/Abbildungen/foo_train_loss_lstm_epochs.pdf')




plt.show()





#https://python4astronomers.github.io/plotting/publication.html
#saving the plot



#print dir and list files in dir
#https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
import os
from natsort import natsorted


mypath_training = "X:\\HandTalkHandtalk\\material\\I6_Gestures\\train" #test
mypath_test = "X:\\HandTalk\\Handtalk\\material\\I6_Gestures\\test"  #training
f = []
for dirpath, dirnames, filenames in os.walk(mypath_training, topdown=False):
    f.extend(filenames)
    break

r = os.walk(mypath_training)

#https://careerkarma.com/blog/python-list-files-in-directory/

n_files = []

for root, directories, files in os.walk(mypath_test, topdown=False):
	#for name in directories:
		#print(os.path.join(root, name))
    print(root)
    files = natsorted(files)
    #print(len(files))
    n_files.append(len(files))

    for name in files:
        #print(name)
		print(os.path.join(root, name))



N_test = sum(n_files)




n_files = []


for root, directories, files in os.walk(mypath_training, topdown=False):
	#for name in directories:
		#print(os.path.join(root, name))
    print(root)
    files = natsorted(files)
    #print(len(files))
    n_files.append(len(files))
    for name in files:
        #print(name)
		print(os.path.join(root, name))

N_training = sum(n_files)

