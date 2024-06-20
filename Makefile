build:
	docker -l debug build -t local/pose_detector_3d .

shell:
	docker run --shm-size 16G --gpus all -it -p 8888:8888 -p 6006:6006 -v /media/tk/data_ext:/root/synthetic_cabin_1m -v /data2:/root/data -v .:/workspace local/pose_detector_3d 

jupyter:
	jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.iopub_data_rate_limit=100000 --NotebookApp.rate_limit_window 100000 --NotebookApp.token=0

jupyter-bg:
	jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.iopub_data_rate_limit=100000 --NotebookApp.rate_limit_window 100000 --no-browser --NotebookApp.token=0 &
