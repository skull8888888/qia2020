import os
import torch
import cv2
from crnn import CRNN
from tqdm import tqdm

import csv
import numpy as np

model = CRNN()
model.load_state_dict(torch.load('55acc.pt'))
model.eval();
model.to('cuda');

data_dir = "qia2020/test/"
emo = {0:'hap', 1:'sur', 2:'neu', 3:'fea', 4:'dis', 5:'ang',6:'sad'}

with open('test_confirm.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['FileID','Emotion'])
    for filename in tqdm(sorted(os.listdir(data_dir))):
        if not filename.endswith(".mp4"): 
            continue 
        
        f = 'torch_video_3_test/' + filename[:5] + '.pt'
        
        X = torch.load(f)
        X = X.unsqueeze(0).to('cuda:0')
        
        with np.load(data_dir + filename[:5] +'.npz') as data:
            T = torch.Tensor(data['word_embed'])
        
        if T.size(0) < 50:
            T = torch.cat([T,torch.zeros(50-T.size(0),200)])

        T = T.unsqueeze(0).to('cuda:0')
        
        y_hat = model((X,T))
        y_pred = y_hat.max(1)[1]

        file_id = filename[:5].strip()
        emotion = emo[y_pred.item()]
        writer.writerow([file_id,emotion])
                  