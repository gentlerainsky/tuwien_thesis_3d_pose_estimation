from modules.lifter_2d_3d.dataset.base_dataset import BaseDataset


class DriveAndActKeypointDataset(BaseDataset):
    def __init__(
        self,
        annotation_file,
        prediction_file,
        bbox_file,
        image_width,
        image_height,
        actors,
        exclude_ankle=False,
        exclude_knee=False,
        is_center_to_neck=False,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=False,
        is_normalize_rotation=None,
        bbox_format='xywh',
        remove_activities=[]
    ):
        super().__init__(
            annotation_file,
            prediction_file,
            bbox_file,
            image_width,
            image_height,
            actors,
            exclude_ankle,
            exclude_knee,
            is_center_to_neck,
            is_normalize_to_bbox,
            is_normalize_to_pose,
            is_normalize_rotation,
            bbox_format,
            remove_activities
        )

    # modified from: https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
    def make_weights_for_balanced_classes(self):
        image_activies = self.image_activities
        activity_types = self.activities
        activity_list = list(activity_types)
        activity_to_id = {activity: idx for idx, activity in enumerate(activity_list)}
        # id_to_activity = {idx: activity for idx, activity in enumerate(activity_list)}
        count = [0] * len(activity_types)
        for activity in image_activies:
            count[activity_to_id[activity]] += 1
        weight_per_class = [0.0] * len(activity_list)
        N = float(sum(count))
        for i in range(len(activity_list)):
            weight_per_class[i] = (N / float(count[i]))
        weight = [0] * len(image_activies)
        for idx, activity in enumerate(image_activies):
            weight[idx] = weight_per_class[activity_to_id[activity]]
        self.sample_weight = weight
