call "%CONDA_ROOT%\Scripts\activate.bat" base
call conda install -c conda-forge -y pdal cython packaging

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

python setup.py build

