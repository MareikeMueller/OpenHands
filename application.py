import omegaconf
from openhands.apis.inference import InferenceModel
import openhands.datasets.isolated
import sys

import imutils
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from scipy import stats


from mediapipe_extract import get_holistic_keypoints





letters=[ "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

cfg = omegaconf.OmegaConf.load("pose_inference_lstm.yaml")
model = InferenceModel(cfg=cfg)

model.init_from_checkpoint_if_available()


#if cfg.data.test_pipeline.dataset.inference_mode:
 #   model.test_inference()
#else:
 #   model.compute_test_accuracy()


#Test in real time

#drawing utilities outsourced in generate_pose.py





from openhands.datasets.pose_transforms import CenterAndScaleNormalize
from openhands.datasets.pose_transforms import PoseSelect
pose = PoseSelect(preset="mediapipe_holistic_minimal_27")
center = CenterAndScaleNormalize(reference_points_preset="shoulder_mediapipe_holistic_minimal_27")


'''
Returns: 
 x (torch.Tensor):  tensor of shape :math:`(N, in_channels, T_{in}, V_{in})`
 N is a batch size, number of files/sequences in batch, since only one sequence at a time is fed to the model 
 N= 1 in adhoc webcam application 
 
 returned tensor forwarded to pose_flattener.py 

'''



def getitem_pose_tensor_adhoc_(dictionary):  #adatptation from base.py returns tensor of size
    """
    Returns
    C - num channels
    T - num frames
    V - num vertices
    """
    data = dictionary
    #data = d
    path = None
    # imgs shape: (T, V, C)
    kps = data["keypoints"]
    scores = data["confidences"]

    # if not self.pose_use_z_axis:
    print("pose_use_z_axis")
    kps = kps[:, :, :2]

    # if self.pose_use_confidence_scores:
    # print("use_confidence")
    # kps = np.concatenate([kps, np.expand_dims(scores, axis=-1)], axis=-1)

    kps = np.asarray(kps, dtype=np.float32)
    data = {
        "frames": torch.tensor(kps).permute(2, 0, 1),  # (C, T, V)
        "label": data["label"],
        "file": path,
        # "lang_code": data["lang_code"] if self.multilingual else None,  # Required for lang_token prepend
        "lang_code": None,
        # "dataset_name": data["dataset_name"] if self.multilingual else None,
        "dataset_name": None,
        # Required to calc dataset-wise accuracy
    }

    # Center and Scale Normalize from pose_transforms
    # if self.transforms is not None:
    # data = self.transforms(data)
    # Center and Scale Normalize from pose_transforms (s. ssl/pose_transforms.py)
    data  = pose.__call__(data)
    data = center.__call__(data)

    #print(data['frames'].shape) #visualize data frames

    batchlist = []
    batchlist.append(data)

    #stack frames # no padding needed since frame number the  same for one sequence

    frames = [data['frames']]
    frames = torch.stack(frames, dim = 0)

    print("new:" , frames)
    labels = [data['label']]
    labels = torch.stack(labels, dim = 0)

    print("old labels", labels )

    #stack labels

    print(frames.shape)

    return frames




#get = getitem_pose_tensor_adhoc_(d) #test data preprocessing (pose transforms, and scaling)



#colors = [(245,117,16), (117,245,16), (16,117,245)]
colors = [(117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16),
       (245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(245,117,16)]
def prob_viz(res, actions_abc, input_frame, colors):
#def prob_viz(res, actions_min_abc, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        print(num)
        #label = equal_label(str(num))
        #print(label)
        number = int(num)
        n_half = num//2
        if number >= 13:
            n = num - 13
            cv2.rectangle(output_frame, (637, 65+n*30), (645 -  int(prob*100), 85+n*30), colors[num], -1)
            cv2.putText(output_frame, actions_abc[num], (620, 80+n*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (211,211,211), 2, cv2.LINE_AA)
            #cv2.putText(output_frame, actions_min_abc[num], (620, 80+n*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        else:

           cv2.rectangle(output_frame, (0,65+num*30), (int(prob*100), 85+num*30), colors[num], -1)
           cv2.putText(output_frame, actions_abc[num], (0, 85+num*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (211,211,211), 2, cv2.LINE_AA)
           #cv2.putText(output_frame, actions_min_abc[num], (0, 85+num*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    return output_frame





#1. New detection variables
sequence=[]
sentence= []
predictions =[]

threshold = 0.3

import torch
import torch.nn.functional as nnf
from openhands.datasets.pipelines.generate_pose import MediaPipePoseGenerator
holistic = MediaPipePoseGenerator()



_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#generate keypoints from web cam frames

cap = cv2.VideoCapture(0)

#frame_width = int(cap.get(3))  ####for video
#frame_height = int(cap.get(4))  ##for video

#size = (frame_width, frame_height)  ####for video

#result = cv2.VideoWriter('in_actionavi',
 #                        cv2.VideoWriter_fourcc(*'MJPG'),
  #                       10, size)  ##########for video

# Set mediapipe model 
#with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
while cap.isOpened():

    # Read feed
    ret, frame = cap.read()


    #drawing utility ######################################
    # Make detections
    image, results = holistic.mediapipe_detection(frame)
    print(results)

    # Draw landmarks
    holistic.draw_styled_landmarks(image, results)



    ###################################################

    # Prediction logic
    sequence.append(frame)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        pose_kps, pose_confs = holistic.get_holistic_keypoints(sequence)
        print(pose_kps)

        body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1)
        confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)
        d = {"keypoints": body_kps, "confidences": confs}
        label = -1
        d['label'] = torch.tensor(label, dtype=torch.long)

        get = getitem_pose_tensor_adhoc_(d) #input tensor, batch size = 1, for each sequence


        y_hat = model(get.to(_device)) # prediction


        prob = nnf.softmax(y_hat, dim=1)
        res = prob.cpu().detach().numpy()[0].tolist()
        conf, classes = torch.max(prob, 1)

        print("confidence: ", conf[0])
        class_indices = torch.argmax(y_hat, dim=-1)
        print(class_indices)

        for i, pred_index in enumerate(class_indices):
            index = pred_index.item()  ##added

            # label = self.datamodule.test_dataset.id_to_gloss[pred_index]
            label = letters[index][0]
            predictions.append(label)
            print(f"{label}")

        if len(sentence)> 0:
            if letters[index] != sentence[-1]:
                sentence.append(letters[index])


        else:
            sentence.append(letters[index])

        if len(sentence) > 5:
            sentence = sentence[-5:]


        #Viz probabilities
        image = prob_viz(res, letters, image, colors)
        # image = prob_viz(res, actions_min_abc, image, colors)

        cv2.rectangle(image, (0, 0), (645, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #result.write(image)  ###for video
        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
#result.release() ###for video
cv2.destroyAllWindows()


#append frames to sequences of length 30 if sequences reached make predictions


#plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, letters, image, colors))
#plt.show()

#prediction
#y_hat = model()
# print(y_hat)

#tensor = torch.tensor(keypoints)
#aufbereiten des dictionaries
#centern der keypoints pose_transforms.py

#y_hat=model(get.to(_device))


#class_indices = torch.argmax(y_hat, dim=-1)
#print(class_indices)


#for i, pred_index in enumerate(class_indices):
 #   index = pred_index.item()  ##added

    # label = self.datamodule.test_dataset.id_to_gloss[pred_index]
  #  label = letters[index]


   # print(f"{label}")

#import pickle
######################
#pose_data = pickle.load(open("X:/HandTalk/Handtalk/material/2_2_1_cam1.pkl", "rb"))

#only one element tensors can be converted to Python scalars
#stackoverflow: how do i get the value  of a tensor in Pytorch
#a = prob.cpu().detach().numpy()[0].tolist()
#enumerate(a)
#np.argmax(a[0])


