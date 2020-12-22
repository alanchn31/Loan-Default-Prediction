FROM panovvv/hadoop-hive-spark:2.5
LABEL maintainer "Alan <alanchn31@gmail.com>"

# COPY . /
# COPY . entrypoint.sh /
# RUN chmod +x /master.sh
# RUN chmod +x /entrypoint.sh

RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools

RUN apk add --no-cache bash git openssh
COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt
COPY . /opt

# Expose port
EXPOSE 8080 7077 8998

CMD ["/bin/bash", "/worker.sh"]