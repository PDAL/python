apt-get update -y; apt-get install python3-pip -y
pip3 install numpy packaging cython
cd /pdal/
python3 setup.py build
python3 setup.py test
