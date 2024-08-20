import torch
from chronos import ChronosPipeline

def load_chronos_pipeline(model_name='amazon/chronos-t5-large', device='cuda', dtype=torch.bfloat16):
    """
    Load the ChronosPipeline model from the pretrained 'amazon/chronos-t5-large' model.

    Parameters:
    model_name (str): Name of the pretrained model. Default is 'amazon/chronos-t5-large'.
    device (str): Device to map the model. Default is 'cuda'.
    dtype (torch.dtype): Data type for the model. Default is torch.bfloat16.

    Returns:
    ChronosPipeline: Loaded ChronosPipeline model.

    Raises:
    Exception: If the model cannot be loaded.
    """
    try:
        chronos_pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
        )
        print(f'Chronos model loaded successfully o/')
        return chronos_pipeline
    except Exception as err:
        print(f'Chronos model cannot be loaded. {err}')
        raise
