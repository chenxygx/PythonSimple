FROM python:3.7
MAINTAINER xychene "xychene@isoftstone.com"
COPY ./requirements.txt /requirements.txt
WORKDIR /
RUN pip install -r requirements.txt
COPY . /
ENTRYPOINT ["python3"]
CMD ["app.py"]