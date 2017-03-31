FROM nvidia/cuda:8.0-cudnn5-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y git openssh-server python3-pip   \
    python-virtualenv gcc gfortran binutils python3-dev libffi-dev libzmq3-dev locate

WORKDIR /
RUN apt-get install -y libopenblas-dev

# compile openblas
#RUN git clone https://github.com/xianyi/OpenBLAS
#RUN cd OpenBLAS && make -j$(nproc) FC=gfortran && make PREFIX=/OpenBLAS install
#RUN echo '/OpenBLAS/lib' > /etc/ld.so.conf.d/openblas.conf && ldconfig

# dependency for numpy install
RUN pip3 install cython

# compile numpy with OpenBLAS support
RUN git clone https://github.com/numpy/numpy
RUN printf "[openblas]\n" \
"libraries = openblas\n" \
"library_dirs = /OpenBLAS/lib\n" \
"include_dirs = /OpenBLAS/include\n" \
"runtime_library_dirs = /OpenBLAS/lib\n" >> numpy/site.cfg


RUN cd numpy && python3 setup.py config && \
    python3 setup.py build -j $(nproc) && python3 setup.py install

# set environment for pip module install
ENV PATH "/OpenBLAS/bin:$PATH"
ENV LD_LIBRARY_PATH "/OpenBLAS/lib:$LD_LIBRARY_PATH"
ENV CUDA_ROOT "/usr/local/cuda"

# create a non-root user
ENV USER user
ENV HOME=/home/${USER}

RUN useradd -s /bin/bash -m ${USER}
# change passwords
RUN echo "${USER}:${USER}" | chpasswd

# ssh configuration
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo 'PermitUserEnvironment yes' >> /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN echo 'export PATH=/OpenBLAS/bin:/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
RUN echo 'export LD_LIBRARY_PATH="/OpenBLAS/lib:$LD_LIBRARY_PATH"' >> /etc/profile.d/cuda.sh
RUN echo 'export CUDA_ROOT=/usr/local/cuda' >> /etc/profile.d/cuda.sh

COPY start.sh start.sh
RUN chmod a+x start.sh
RUN mkdir /var/run/sshd

EXPOSE 22
CMD ["/start.sh"]

ENV APP_HOME /home/user
WORKDIR $APP_HOME

COPY requirements.txt requirements.txt

RUN pip3 install -U pip setuptools wheel
RUN pip3 install -r requirements.txt
RUN mkdir models
