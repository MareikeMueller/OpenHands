import csv
import csv
import numpy as np
import pandas as pd
import os

path = "X:/HandTalk/Handtalk/material/I6_Gestures/video"
path_s = "X:/HandTalk/Handtalk/material/I6_Gestures/"

#create csv with video_name, and tag


#create an empty Dataframe with column names
dir = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']

def get_label(d):
    directories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z']

    index = dir.index(d)

    return directories[index]



column_names = ['video_name', 'tag']
df = pd.DataFrame(columns=column_names)




for d in dir:

    directory = os.path.join(path, d)
    tag = get_label(d)
    videos = os.listdir(directory)

    #sort numeric strings in a list
    res = sorted(videos, key=lambda x: (len(x), x))

    for i in range(80): #2 recordings of every 'signer', 20 signers, 2 camera views
        row = {'video_name': res[i], 'tag': tag}
        df = df.append(row, ignore_index=True)





save_path = 'X:/HandTalk/Handtalk/material/I6_Gestures/data.csv'

df.to_csv(save_path, index = False)



#split test set in train and test set
df2 =pd.read_csv(save_path)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.25, random_state=42, shuffle = True)



#https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213

train, validate, test = np.split(df2.sample(frac=1, random_state=42),[int(.6 * len(df2)), int(.8 * len(df2))])



train_path = os.path.join(path_s, 'train.csv')
test_path= os.path.join(path_s, 'test.csv')
val_path = os.path.join(path_s, 'val.csv')





#save train and test file
train.to_csv(train_path, index = False)
test.to_csv(test_path, index = False)
validate.to_csv(val_path, index = False)



#https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
#https://towardsdatascience.com/how-to-split-a-dataframe-into-train-and-test-set-with-python-eaa1630ca7b3

#aufteilen der files in die ordner test und train

path_test = 'X:/HandTalk/Handtalk/material/I6_Gestures/test'
path_train = 'X:/HandTalk/Handtalk/material/I6_Gestures/train'
path_val = 'X:/HandTalk/Handtalk/material/I6_Gestures/val'

df_test = pd.read_csv(test_path)
df_train= pd.read_csv(train_path)



#create an empty Dataframe with column names
dir = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']

import shutil




for d in dir:

    directory = os.path.join(path, d)
    videos = os.listdir(directory)

    #sort numeric strings in a list
    res = sorted(videos, key=lambda x: (len(x), x))

    for i in range(80): #2 recordings of every 'signer', 20 signers, 2 camera views

        if res[i] in df_test['video_name'].tolist():
            os.makedirs(os.path.join(path_test, get_label(d)), exist_ok=True) # added
            src = os.path.join(directory, res[i])
            dst = os.path.join(path_test, get_label(d), res[i])
            shutil.copyfile(src, dst)
        if res[i] in df_train['video_name'].tolist():
            os.makedirs(os.path.join(path_train, get_label(d)), exist_ok=True) #added
            src = os.path.join(directory, res[i])
            dst = os.path.join(path_train, get_label(d), res[i])
            shutil.copyfile(src, dst)
        else:
            os.makedirs(os.path.join(path_val, get_label(d)), exist_ok=True)  #added whole else statement for validation data set
            src = os.path.join(directory, res[i])
            dst = os.path.join(path_val, get_label(d), res[i])
            shutil.copyfile(src, dst)





#r= df_test['video_name']
