import pandas as pd

# 텍스트 파일 읽기
txt_file = "/data/jongseo/project/cil/videoCIL/data/HMDB51/testlist01.txt"  # TXT 파일 경로
csv_file = "/data/jongseo/project/cil/videoCIL/data/HMDB51/test.csv"  # 저장할 CSV 파일 경로

# 데이터를 불러와 공백 기준으로 분할
df = pd.read_csv(txt_file, sep=" ", header=None)

# 열 이름 지정 (예: 'filename'과 'label'로 지정)
df.columns = ["filename", "label"]

# CSV 파일로 저장
df.to_csv(csv_file, index=False)

print(f"CSV 파일이 저장되었습니다: {csv_file}")