# DUSt3R-to-COLMAP
Pulling data from DUSt3R to fit the COLMAP output format.

### Preparation
```
wget -nc https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

git clone https://github.com/naver/dust3r --recursive

pip install -r requirements.txt
```
Make dir 'scene_name' in data/images/
Download image to scene_name folder

### Run Code
Directory setting is needed before running the code

```
dust3r_inference.py
```

### Data directory
![image](https://github.com/user-attachments/assets/28e5fa32-6d62-4d64-a57a-544d19943680)

- Source images are saved under `/data/images` with each folder named after the `scene_name`.  
- For each `scene_name`, a directory `/data/scenes/scene_name` is created, and inside it, the required COLMAP output data is generated in the `sparse/0` folder.
