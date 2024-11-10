# DUSt3R-to-COLMAP
Pulling data from DUSt3R to fit the COLMAP output format.

### Preparation
```
wget -nc https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive

git clone https://github.com/naver/dust3r --recursive

pip install -r requirements.txt
```
data/images/내에 scene_name 폴더 만들고 이미지 넣어두기

### 실행
dust3r_inference.py 내에서 경로 설정 -> 실행

```
dust3r_inference.py
```

### 데이터
![image](https://github.com/user-attachments/assets/28e5fa32-6d62-4d64-a57a-544d19943680)

* source 이미지를 `/data` 내 `/images ` 안에 각 scene_name으로 저장
* `/data` 내 `/scenes/scene_name `이 생성되고 안에 `sparse/0`에 필요한 COLMAP 아웃풋 형태의 데이터가 생성됨
  