build:
	docker -l debug build -t local/pose_detector_3d .
	# docker build -t mmpose docker/

shell:
	# docker run --shm-size 16G --gpus all -it -p 8888:8888 -v ~/Documents/Projects/data:/workspace/data -v .:/workspace local/synthetic_cabin
	docker run --shm-size 16G --gpus all -it -p 8888:8888 -p 6006:6006 -v /data2:/root/data -v .:/workspace local/pose_detector_3d
	# docker run --shm-size 16G --gpus all -it -p 8888:8888 -v /data2:/root/data -v .:/workspace local/pose_detector_3d

jupyter:
	jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.iopub_data_rate_limit=100000 --NotebookApp.rate_limit_window 100000

jupyter-bg:
	jupyter notebook --quiet --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.iopub_data_rate_limit=100000 --NotebookApp.rate_limit_window 100000 --no-browser &

tensorboard:
	# make tensorboard LOG_DIR="mmengine_workdir/pose_estimator_2d/20230802_135548/vis_data/"
	tensorboard --bind_all --logdir=$(LOG_DIR) --port 6006

wandb:
	wandb login d1d1f1554683acae95405d133a4f608c3757c72c