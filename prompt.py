# from models import clip
from models.clip_custom import clip
import torch
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import json
import pickle

def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id


def text_prompt(dataset='HMDB51', data_path = None ,clipbackbone='ViT-B/16', device='cpu', text_finetune=None,args=None):
    actionlist, actionprompt, actiontoken = [], {}, []
    numC = {'HMDB51-feature-30fps-center': 51,}

    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    if text_finetune is not None:
        import torch.nn as nn
        clipmodel.text_projection = nn.Parameter(torch.zeros(512, 512//2))
        clipmodel.image_projection = nn.Parameter(torch.zeros(768, 512//2))
        lavila = torch.load(text_finetune, map_location='cpu')
        lavila_checkpoint = lavila['state_dict']
        new_dict = clipmodel.state_dict()
        for key in lavila_checkpoint: #allkeys들은 모두 module.으로 시작한다. visual부분을 빼기위해서 
            if not key.startswith('module.visual'):
                new_dict[key[7:]] = lavila_checkpoint[key]
        # load로 불러온 pre-trained weight를 new_dict에 담아주고
        clipmodel.load_state_dict(new_dict)
        clipmodel.to(device)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False
    clipmodel.to(device)
    clipmodel.eval()
    # for paramclip in clipmodel.parameters():
    #     paramclip.requires_grad = False

    # convert to token, will automatically padded to 77 with zeros
    if dataset == 'HMDB51-feature-30fps-center':
        meta = open("../data/HMDB51/HMDB51_action.list", 'rb')
        actionlist = meta.readlines()
        meta.close()
        actionlist = np.array([a.decode('utf-8').split('\n')[0] for a in actionlist])
        actiontoken = np.array([convert_to_token(a) for a in actionlist])
    # More datasets to be continued

    elif dataset == 'EPIC':
        noun_anno_path = os.path.join(data_path, 'epic100_noun_classes.csv')
        verb_anno_path = os.path.join(data_path, 'epic100_verb_classes.csv')
        noun_cleaned = pd.read_csv(noun_anno_path, header=None, delimiter=',')
        verb_cleaned = pd.read_csv(verb_anno_path, header=None, delimiter=',')
        nounlist = list(noun_cleaned.values[:, 0])
        verblist = list(verb_cleaned.values[:, 0])
        nountoken = np.array([convert_to_token(a) for a in nounlist])
        verbtoken = np.array([convert_to_token(a) for a in verblist])
    
        # query the vector from dictionary
        with torch.no_grad():
            nounembed = clipmodel.encode_text_light(torch.tensor(nountoken).to(device))
            verbembed = clipmodel.encode_text_light(torch.tensor(verbtoken).to(device))

        noundict = OrderedDict((nounlist[i], nounembed[i].cpu().data.numpy()) for i in range(300))
        verbdict = OrderedDict((verblist[i], verbembed[i].cpu().data.numpy()) for i in range(97))
        nountoken = OrderedDict((nounlist[i], nountoken[i]) for i in range(300))
        verbtoken = OrderedDict((verblist[i], verbtoken[i]) for i in range(97))

        return [nounlist, noundict, nountoken, verblist, verbdict, verbtoken]
    elif dataset == 'ActivityNet':
        with open(data_path, 'rb') as file:
            # 피클 파일에서 객체 로드
            pkl = pickle.load(file)
        class_per_task =args.nb_classes//args.num_tasks
        cil_actionlist = []
        cil_actiondict =[]
        cil_actiontoken =[]
        all_actionlist = []
        for task_id in range(args.num_tasks):
            actionlist=(list(pkl['train'][task_id].keys()))
            class_per_task = len(actionlist)

            all_actionlist+=actionlist
            #action_label is like nounlist.
            actiontoken = np.array([convert_to_token(a) for a in actionlist])
            with torch.no_grad():
                actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))
            actiondict = OrderedDict((actionlist[i],actionembed[i].cpu().data.numpy()) for i in range(class_per_task))
            actiontoken = OrderedDict((actionlist[i],actiontoken[i]) for i in range(class_per_task))
            cil_actionlist.append(actionlist)
            cil_actiondict.append(actiondict)
            cil_actiontoken.append(actiontoken)
        all_actiontoken = np.array([convert_to_token(a) for a in all_actionlist])
        with torch.no_grad():
            all_actionembed = clipmodel.encode_text_light(torch.tensor(all_actiontoken).to(device))
        all_actiondict = OrderedDict((all_actionlist[i],all_actionembed[i].cpu().data.numpy()) for i in range(args.nb_classes))
        all_actiontoken = OrderedDict((all_actionlist[i],all_actiontoken[i]) for i in range(args.nb_classes))
        all_class_list = (all_actionlist,all_actiondict,all_actiontoken)
        return cil_actionlist,cil_actiondict,cil_actiontoken, all_class_list
    elif dataset == 'SSV2':
        with open(data_path, 'rb') as file:
            # 피클 파일에서 객체 로드
            pkl = pickle.load(file)
        # class_per_task =args.nb_classes//args.num_tasks
        cil_actionlist = []
        cil_actiondict =[]
        cil_actiontoken =[]
        all_actionlist = []
        for task_id in range(args.num_tasks):
            actionlist=(list(pkl['train'][task_id].keys()))
            class_per_task = len(actionlist)

            all_actionlist+=actionlist
            #action_label is like nounlist.
            actiontoken = np.array([convert_to_token(a) for a in actionlist])
            with torch.no_grad():
                actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))
            actiondict = OrderedDict((actionlist[i],actionembed[i].cpu().data.numpy()) for i in range(class_per_task))
            actiontoken = OrderedDict((actionlist[i],actiontoken[i]) for i in range(class_per_task))
            cil_actionlist.append(actionlist)
            cil_actiondict.append(actiondict)
            cil_actiontoken.append(actiontoken)
    elif dataset == 'UCF101':
        with open(data_path, 'rb') as file:
            # 피클 파일에서 객체 로드
            pkl = pickle.load(file)
        # class_per_task =args.nb_classes//args.num_tasks
        cil_actionlist = []
        cil_actiondict =[]
        cil_actiontoken =[]
        all_actionlist = []
        for task_id in range(args.num_tasks):
            actionlist=(list(pkl['train'][task_id].keys()))
            class_per_task = len(actionlist)

            all_actionlist+=actionlist
            #action_label is like nounlist.
            actiontoken = np.array([convert_to_token(a) for a in actionlist])
            with torch.no_grad():
                actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))
            actiondict = OrderedDict((actionlist[i],actionembed[i].cpu().data.numpy()) for i in range(class_per_task))
            actiontoken = OrderedDict((actionlist[i],actiontoken[i]) for i in range(class_per_task))
            cil_actionlist.append(actionlist)
            cil_actiondict.append(actiondict)
            cil_actiontoken.append(actiontoken)
            
        #전체 용 만들기
        all_actiontoken = np.array([convert_to_token(a) for a in all_actionlist])
        with torch.no_grad():
            all_actionembed = clipmodel.encode_text_light(torch.tensor(all_actiontoken).to(device))
        all_actiondict = OrderedDict((all_actionlist[i],all_actionembed[i].cpu().data.numpy()) for i in range(args.nb_classes))
        all_actiontoken = OrderedDict((all_actionlist[i],all_actiontoken[i]) for i in range(args.nb_classes))
        all_class_list = (all_actionlist,all_actiondict,all_actiontoken)
        return cil_actionlist,cil_actiondict,cil_actiontoken, all_class_list
        
        
    # query the vector from dictionary
    with torch.no_grad():
        actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))

    actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(numC[dataset]))
    actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(numC[dataset]))

    return actionlist, actiondict, actiontoken