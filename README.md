# Setup Instructions for evaluating D3VO on Tera Internal Flights
```bash
# clone the repo
git clone https://github.com/deep-gabani/D3VO.git
cd D3VO

# create conda env
conda create -n d3vo python=3.8.10
conda activate d3vo

# install necessary libraries
apt-get install libsuitesparse-dev ffmpeg
python3 -m pip install torch torchvision numpy pandas scipy matplotlib natsort opencv-python
conda install -c conda-forge suitesparse
export LD_LIBRARY_PATH=/root/miniconda3/envs/d3vo/lib:$LD_LIBRARY_PATH


# download trained DepthNet and PoseNet weights (from the [drive](https://drive.google.com/drive/folders/176fuEVP1BVQlKQNXCp3wQE_kBK_ogOCT))
mkdir weights && cd weights
wget \
    https://storage.googleapis.com/tera-public/algorithms-analysis-for-benchmarking/d3vo/weights/adam.pth \
    https://storage.googleapis.com/tera-public/algorithms-analysis-for-benchmarking/d3vo/weights/depth.pth \
    https://storage.googleapis.com/tera-public/algorithms-analysis-for-benchmarking/d3vo/weights/encoder.pth \
    https://storage.googleapis.com/tera-public/algorithms-analysis-for-benchmarking/d3vo/weights/pose.pth
cd ..

# compile and install
cd g2opy
mkdir build && cd build
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 cmake ..
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 make -j8
cd ..
python setup.py install
cd ..

# Now you can go to poc.ipynb notebook and run D3VO on flights.
```


# D3VO

16-833 Robot Localization and Mapping course project

Implementation of D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry (https://arxiv.org/pdf/2003.01060.pdf)



## Setup 

- Use Python 3.8.10 or earlier.

- Compile and install g2opy.
```
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install
cd ..
```

- Download trained DepthNet and PoseNet weights. https://drive.google.com/drive/folders/176fuEVP1BVQlKQNXCp3wQE_kBK_ogOCT?usp=sharing


- Install Python packages.

```
python3 -m pip install torch torchvision numpy matplotlib opencv-python
```

- (Optional) Download and install the KITTI color odoometry dataset and convert frames to a video format (like MP4). Optionally also download ground truth poses for evaluation. https://www.cvlibs.net/datasets/kitti/eval_odometry.php


- Run D3VO on an input video (.mp4). 
```
python3 src/main.py video_path.mp4 weights_directory_path --gt optional_ground_truth_txt_path --out output_dir_path
```



## Sources

- g2opy: https://github.com/uoip/g2opy

- monodepth2: https://github.com/nianticlabs/monodepth2

- kitti-odom-eval: https://github.com/Huangying-Zhan/kitti-odom-eval

