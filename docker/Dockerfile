FROM ubuntu:20.04

RUN apt update && apt -y upgrade
RUN apt install -y python3 pip
RUN DEBIAN_FRONTEND=noninteractive TZ=8/36 apt install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng
RUN DEBIAN_FRONTEND=noninteractive TZ=8/36 apt install -y cm-super

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt