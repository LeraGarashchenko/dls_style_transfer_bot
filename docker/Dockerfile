FROM python:3.10-slim-buster as bulilder
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install -r requirements.txt

FROM python:3.10-slim-buster
WORKDIR /code
ENV PYTHONPATH=/code
COPY --from=bulilder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY style_transfer_bot /code/style_transfer_bot
CMD ["python", "style_transfer_bot/bot.py"]