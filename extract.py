import os
import decord
from decord import VideoReader
from decord import cpu
import cv2

def resize_frame(frame, new_short):
    h, w, _ = frame.shape
    if h > w:
        new_h = int(h * new_short / w)
        new_w = new_short
    else:
        new_w = int(w * new_short / h)
        new_h = new_short
    return cv2.resize(frame, (new_w, new_h))

def extract_frames(video_path, output_dir, new_short=None):
    # VideoReader를 사용하여 비디오를 읽습니다.
    vr = VideoReader(video_path, ctx=cpu(0))

    # 출력 디렉터리가 없으면 생성합니다.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모든 프레임을 추출하고 저장합니다.
    for i in range(len(vr)):
        frame = vr[i].asnumpy()
        if new_short:
            frame = resize_frame(frame, new_short)
        output_file_path = os.path.join(output_dir, f'img_{i + 1:05d}.jpg')
        cv2.imwrite(output_file_path, frame)
    print(f'{video_path} done! frame length : {len(vr)}')
def process_all_videos_from(video_dir, output_base_dir, new_short=None, start_from_file='v_zzxYEZkahBU.mp4'):
    video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    
    # 이미 처리된 파일 이후의 파일만을 포함하도록 리스트를 수정합니다.
    if start_from_file in video_files:
        start_idx = video_files.index(start_from_file) + 1
    else:
        start_idx = 0

    for video_file in video_files[start_idx:]:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_base_dir, video_name)
        extract_frames(video_path, output_dir, new_short)

# 사용 예:
video_dir = '/data2/local_datasets/ActivityNet/videos'
output_base_dir = '/data2/local_datasets/ActivityNet/rawframes'
new_short_value = 256  # 원하는 값으로 설정
process_all_videos_from(video_dir, output_base_dir, new_short_value)
