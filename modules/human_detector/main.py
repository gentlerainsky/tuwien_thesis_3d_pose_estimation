from modules.human_detector.human_detector import HumanDetector
import pathlib
import json
import datetime
import os


def finetune():
    human_detector = HumanDetector(
        config_path='./src/modules/human_detector/config/faster_rcnn.py',
        pretrained_path='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'\
            'faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth',
        checkpoint_path='./mmengine_workdir/human_detector/epoch_1.pth',
        data_root_path='/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/',
        device='cuda:0',
        working_directory='./mmengine_workdir/human_detector',
        log_level='CRITICAL'
    )
    human_detector.finetune()


def detect_person(
    config_path='./src/modules/human_detector/config/faster_rcnn.py',
    pretrained_path='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'\
        'faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth',
    checkpoint_path='./mmengine_workdir/human_detector/epoch_1.pth',
    dataset_root_path='/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/',
    device='cuda:0',
    working_directory='./mmengine_workdir/human_detector',
    log_level='CRITICAL'
):
    dataset_root = pathlib.Path(dataset_root_path)
    human_detector = HumanDetector(
        config_path=config_path,
        pretrained_path=pretrained_path,
        checkpoint_path=checkpoint_path,
        data_root_path=dataset_root.as_posix(),
        device=device,
        working_directory=working_directory,
        log_level=log_level
    )
    human_detector.load_pretrained()
   
    image_sets = ['train', 'val', 'test']
    for image_set in image_sets:
        results = []
        image_folder = dataset_root / 'images' / image_set
        image_paths = list(image_folder.iterdir())
        count = 0
        max_count = len(image_paths)
        start_time = datetime.datetime.now()
        print(f'\timage_set = {image_set}, start at = {start_time}')
        for image_path in image_paths:
            img_path_str = str(image_path)
            detector_result = human_detector.get_bbox(img_path_str)
            bboxes = detector_result['bboxes']
            scores = detector_result['scores']
            if bboxes.shape[0] == 0:
                print(f'problematic images = {img_path_str}')
                continue
            result = {
                'image_id': int(image_path.name.split('.')[0]),
                'category_id': 1,
                'bbox': bboxes.round().cpu().numpy()[0].tolist(),
                'score': scores.cpu().numpy().astype(float).round(decimals=4)[0]
            }
            results.append(result)
            count += 1
            if (count + 1) % 500 == 0:
                print(
                    f'\t\ttime = {datetime.datetime.now()}\n'
                    f'\t\t\tprogress = {count + 1}/{max_count} '
                        f'= {(count + 1)/max_count:.4f}'
                )

            # if (count + 1) % 100 == 0:
            #     break
        end_time = datetime.datetime.now()
        total_secs = (end_time - start_time).total_seconds()
        total_mins = total_secs / 60
        print(f'\t\timage_set = {image_set}, finish at = {datetime.datetime.now()}')
        print(f'\t\ttime spend = {total_mins:.2f} mins ({total_secs:.2f} secs), fps = {(count + 1)/total_secs:.4f} fps')
        
        result_folder = dataset_root / 'person_detection_results'
        if not result_folder.exists():
            print(f'created a folder at {str(result_folder)}.')
            os.makedirs(str(result_folder))
        with (result_folder / f'human_detection_{image_set}.json').open('w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    pass
