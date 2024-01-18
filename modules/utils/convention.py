import os


def get_saved_model_path(
    model_name,
    model_suffix,
    trained_dataset_name,
    trained_datasubset_name=None,
    pretrained_dataset_name=None,
    pretrained_datasubset_name=None,
    create_if_not_exists=True
):
    saved_model_path = f'./saved_lifter_2d_3d_model/{model_name}/{trained_dataset_name}'

    if trained_datasubset_name is not None:
        saved_model_path += f'/{trained_datasubset_name}'

    if pretrained_dataset_name is not None:
        saved_model_path += f'/transfer_learning/{pretrained_dataset_name}'
        if pretrained_datasubset_name is not None:
            saved_model_path += f'/{pretrained_datasubset_name}'

    saved_model_path += f'/{model_name}'
    
    if model_suffix is not None:
        saved_model_path = f'{model_suffix}'

    if create_if_not_exists:
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)

    return saved_model_path
