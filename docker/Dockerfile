FROM ghcr.io/timeeval/python3-torch:0.3.0

LABEL maintainer="arja.anton@gmail.com"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# add algorithm implementation
COPY algorithm.py /app/