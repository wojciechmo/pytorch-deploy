# Deploy PyTorch ResNet18 model trained with Python using C++.

Develop using PyTorch Python API:
1. create virtualenv and install dependencies
virtualenv --system-site-packages -p python3.6 ./venv && source ./venv/bin/activate
pip install torch torchvision opencv-python
2. train and save model converted to Torch Script:
python train.py

Deploy using PyTorch C++ API:
1. install PyTorch:
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip -d /tmp
2. install OpenCV:
wget https://github.com/opencv/opencv/archive/3.4.6.zip
unzip opencv-3.4.6.zip -d /tmp
cd /tmp/opencv-3.4.6
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/tmp/opencv-3.4.6/install ..
make -j8
sudo make install
3. build and run c++ program:
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="/tmp/libtorch;/tmp/opencv-3.4.6/install" ..
make
./app
