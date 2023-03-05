# Install recent versions build tools, including Clang and libc++ (Clang's C++ library)
sudo apt install -y clang libc++-dev libc++abi-dev cmake ninja-build

# Install libraries for image I/O and the graphical user interface
sudo apt install -y libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev

# Install required Python packages
sudo apt install -y python3-dev python3-distutils python3-setuptools

# For running tests
sudo apt install -y python3-pytest python3-pytest-xdist python3-numpy

# For generating the documentation
sudo apt install -y python3-sphinx python3-guzzle-sphinx-theme python3-sphinxcontrib.bibtex
