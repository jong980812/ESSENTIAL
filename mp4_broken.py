import os
import os.path as osp
import subprocess
import pandas as pd
import csv
from tqdm import tqdm
#! 이 폴더 하위 존재하는 mp4의 유효성 체크
video_root='/local_datasets/Activitynet/videos'
videos = os.listdir(video_root)
videos = set(videos)
print(f'all samples num: {len(videos)}')
print("STARTR")
for i, k in tqdm(enumerate(videos)):
    # if i% 500 == 0:#! 현재 몇개까지 했는지 알기i 위해.
    #     print(f"{i} checking")
        
    video_path = osp.join(video_root, k)
    command = [
        'ffmpeg', '-v','error',
        '-i','"%s"' % video_path,
        '-f', 'null','-'
    ]
    command = ' '.join(command)
    try:
        subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(
            f'{k} of Video Failed',
            flush=True)#!이 메시지가 뜨면 깨진 파일임.