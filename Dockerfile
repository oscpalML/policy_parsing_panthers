FROM python:3

ADD script.py /script.py
ADD requirements.txt /requirements.txt

ADD oscpalML/XLM-RoBERTa-political-classification /oscpalML/XLM-RoBERTa-political-classification

RUN pip3 install -r /requirements.txt

ENTRYPOINT [ "python3", "/script.py", "-i", "$inputDataset", "-o", "$outputDir" ]

