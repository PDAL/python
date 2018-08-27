apt-get update -y; apt-get install python-pip -y
pip install numpy packaging cython
cd /pdal/
python setup.py build
python setup.py test
