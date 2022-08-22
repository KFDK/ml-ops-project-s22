# first stage
FROM nvidia/cuda:11.7.1-base-ubuntu22.04 as builder
RUN apt-get update && apt-get install -y curl wget gcc build-essential

# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# create env with python 3.5
RUN /opt/conda/bin/conda create -y -n myenv python=3.7.10

# install requirements
WORKDIR /app
COPY requirements.txt /app
ENV PATH=/opt/conda/envs/myenv/bin:$PATH    
RUN pip install -r requirements.txt
RUN pip install cloudml-hypertune
RUN pip uninstall -y pip



####################
# second stage (note: FROM container must be the same as builder)
FROM nvidia/cuda:11.7.1-base-ubuntu22.04 as runner

RUN apt-get update && apt-get install -y curl wget gcc build-essential

# copy environment data including python
COPY --from=builder /opt/conda/envs/myenv/bin /opt/conda/envs/myenv/bin
COPY --from=builder /opt/conda/envs/myenv/lib /opt/conda/envs/myenv/lib

# do some env settings
ENV PATH=/opt/conda/envs/myenv/bin:$PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN echo $PATH 


#RUN echo $PYTHONPATH
####################
# final image
FROM runner
WORKDIR /root   
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY configs/ configs/
COPY models/ models/
ENTRYPOINT [ "python", "src/models/train_model.py"]