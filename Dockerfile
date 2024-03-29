FROM kernai/refinery-parent-images:v1.14.0-torch-cpu

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/run.sh"]