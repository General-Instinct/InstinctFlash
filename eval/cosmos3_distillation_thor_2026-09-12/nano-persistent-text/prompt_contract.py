"""Inspect the actual native prompt transform, not the unrelated policy config."""
def verify_plain_prompt(service):
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    transform = service._transform
    if not isinstance(transform, ActionTransformPipeline):
        raise ValueError('Require the native action transform pipeline')
    if transform.prompt_json_formatter is not None:
        raise ValueError('Nano requires the native plain-text prompt pipeline')
    augmentors = ('viewpoint_augmentor', 'duration_fps_augmentor', 'resolution_info_augmentor')
    if any(getattr(transform, name) is None for name in augmentors):
        raise ValueError('Native Nano prompt metadata augmentors must remain enabled')
    if transform.max_action_dim != 64 or service.cfg.history_length != 1 or service.cfg.action_chunk_size != 32:
        raise ValueError('Require native Nano history1/future32 action layout')
    return dict(format_prompt_as_json=False, metadata_augmentors=list(augmentors),
                max_action_dim=64, history_length=1, action_chunk_size=32,
                observed='service._transform')
