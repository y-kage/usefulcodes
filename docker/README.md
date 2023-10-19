# Before Starting
Modify files.


## .env
Change uid, gid, workdir, ports.
```bash
id -u # UID
id -g # GID
```


## docker-compose.yaml
Change image, container_name, volumes, shm_size if needed.
- image : name of image cached to local. if exist, load, if not, build.
- container_name : name used to get in the container.
- volumes : Correspondence. {local_dir}:{container_dir}
- shm_size : shared memory size. check your spec.


## Dockerfile
Change docker image, libraries, python version.


# Commands

Docker compose up
```bash
cd docker
docker-compose up -d
```

If Dockerfile changed, Docker compose up with build
```bash
cd docker
docker-compose up -d --build
```

Execute command in Docker
```bash
docker exec -it {container_name} bash
docker exec -it -w {WORK_DIR_PATH} {container_name} bash
```

As root
```bash
docker exec -it -u 0 -w {WORK_DIR_PATH} {container_name} bash
```

Using JupyterLab (Optional)
```bash
python -m jupyterlab --ip 0.0.0.0 --port {CONTAINER_PORT} --allow-root
```

Using Tensorboard
```bash
tensorboard --logdir=/workspace/PytorchLightning/lightning_logs --host=0.0.0.0 --port={CONTAINER_PORT}
python /home/{USER}/.local/lib/python3.9/site-packages/tensorboard/main.py --logdir=/workspace/PytorchLightning/lightning_logs --host=0.0.0.0 --port={CONTAINER_PORT}
```

Login W & D
```bash
wandb login
python3 -m wandb login
/usr/bin/python3 -m wandb login
```
