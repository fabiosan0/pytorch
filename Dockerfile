FROM python:3.9

ADD main.py .

ADD AAPL.csv .

RUN pip install pandas matplotlib torch

CMD ["python","./main.py"]