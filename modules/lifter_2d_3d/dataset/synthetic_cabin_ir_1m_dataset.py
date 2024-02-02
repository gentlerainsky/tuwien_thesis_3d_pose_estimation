from modules.lifter_2d_3d.dataset.base_dataset import BaseDataset


class SyntheticCabinIR1MKeypointDataset(BaseDataset):
    def __init__(
        self,
        annotation_file,
        prediction_file,
        bbox_file,
        image_width,
        image_height,
        actors=None,
        exclude_ankle=False,
        exclude_knee=False,
        is_silence=True,
        is_center_to_neck=False,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=False,
        is_normalize_rotation=None,
        bbox_format='xywh',
        remove_activities=[],
        included_view=None,
        is_gt_2d_pose=False
    ):
        self.included_view = included_view
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
            is_silence,
            is_normalize_to_bbox,
            is_normalize_to_pose,
            is_normalize_rotation,
            bbox_format,
            remove_activities,
            is_gt_2d_pose=is_gt_2d_pose
        )

    def filter_samples(self, images):
        if self.included_view is not None:
            results = filter(
                lambda x: x['view'] in self.included_view, images
            )
            return results
        return images
