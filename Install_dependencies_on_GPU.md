1. Your GPU needs to have a compute capability of 3.0 or higher (Check here https://developer.nvidia.com/cuda-gpus)

2. Install Cuda 8 or 9 according to your gpu.
Follow this guide to install : https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
Check also if you have the correct requirements. (Kernel, drivers etc.)
Be carrefull to proceed the post-installation actions, particularly environment setup.
You have to activate POWER8 (Cuda 8)/ POWER9 (Cuda 9) (It's explainedhow to proceed in the installation guide)

3. Install CuDNN 
https://developer.nvidia.com/cudnn
You have to create an account in order to download the setup file.
Choose the correct version according to your cuda version. (Personnaly i have cuda 8.0 and CuDNN 6)

4. Install Dlib by following this guide https://www.learnopencv.com/install-dlib-on-ubuntu/

5. For the other dependencies, follow our guide "Install_depencies"
