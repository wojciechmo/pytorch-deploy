# Deploy PyTorch ResNet18 model trained with Python using C++.

Develop using PyTorch Python API:
1. create virtualenv and install dependencies:<br/>
virtualenv --system-site-packages -p python3.6 ./venv && source ./venv/bin/activate<br/>
pip install torch torchvision opencv-python<br/>
2. train and save model converted to Torch Script:<br/>
python train.py<br/>

Deploy using PyTorch C++ API:
1. install PyTorch:<br/>
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip<br/>
unzip libtorch-shared-with-deps-latest.zip -d /tmp<br/>
2. install OpenCV:<br/>
wget https://github.com/opencv/opencv/archive/3.4.6.zip<br/>
unzip opencv-3.4.6.zip -d /tmp<br/>
cd /tmp/opencv-3.4.6<br/>
mkdir build && cd build<br/>
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/tmp/opencv-3.4.6/install ..<br/>
make -j8<br/>
sudo make install<br/>
3. build and run c++ program:<br/>
mkdir build && cd build<br/>
cmake -DCMAKE_PREFIX_PATH="/tmp/libtorch;/tmp/opencv-3.4.6/install" ..<br/>
make<br/>
./app<br/>
