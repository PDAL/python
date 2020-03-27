FROM continuumio/miniconda3:latest
MAINTAINER Howard Butler <howard@hobu.co>

RUN apt-get install -y \
        gdb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda update -n base -c defaults conda \
    && conda install -y -c conda-forge \
    compilers \
    pdal \
    make ninja \
    python=3.8


RUN git clone https://github.com/PDAL/python.git pdal-python \
    && cd pdal-python \
    && git checkout remove-python-from-pdal-base

RUN cd pdal-python \
    && pip install -e .

ENV PDAL_DRIVER_PATH=/pdal-python/_skbuild/linux-x86_64-3.8/cmake-install/lib/



#gdb --args /pdal-python/_skbuild/linux-x86_64-3.8/cmake-build/pdal_filters_python_test



