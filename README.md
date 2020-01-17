## Deploying with Docker
You will need docker and `nvidia-container-toolkit` on your system to use the `--gpus` option.

```
docker build -t questionable:latest .
docker run --cpus 4 --gpus all -it -p 8765:8765 -v $PWD/cache:/cache -v $PWD/data:/data questionable:latest
```

Starting for the first time can take a while since it downloads SQUADv2, BERT pretrained models, and builds indices.
You can run the commands `/app/entrypoint.sh (prepare|index|serve)` on the container manually.

To refresh the index run with the `--force` flag:

```
/app/entrypoint.sh index --force
```

## Docker compose
Unfortunately, Docker Compose version 3 does not support `runtime` or `gpus` options. You can still use the `runtime` option
with version 2.3 and 2.4, however. You will need to install `nvidia-docker2`.

```
docker-compose up
```
