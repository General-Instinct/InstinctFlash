"""Offline cross-request extension of the pinned Cosmos conditioning cache."""
import torch
from cosmos3_iwm.conditioning_cache import ConditioningCache


def model_signature(model):
    # Version counters are required: an untracked mutable inference tensor cannot
    # establish validity across requests merely from its address.
    weights=[]
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        try:
            version=value._version
        except RuntimeError as error:
            raise ValueError('Persistent cache requires tracked parameter versions') from error
        weights.append((name,id(value),value.data_ptr(),version,tuple(value.shape),
                        tuple(value.stride()),value.dtype,value.device))
    return (tuple(weights),tuple((name,module.training) for name,module in model.named_modules()),
            torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)


class PersistentTransactions:
    """Generation-level publication and invalidation; module reuse stays in parent."""
    def persistent_init(self):
        self.published=set()
        self.signature=None
        self.persistent_stats=dict(commits=0,aborts=0,invalidations=0)

    def invalidate(self):
        self.slots.clear()
        self.seen.clear()
        self.published.clear()

    def velocity(self, original, **kwargs):
        tokens=kwargs.get('text_tokens')
        if tokens is not None:
            key=(tuple(tuple(x) for x in tokens),bool(kwargs.get('skip_text_tokens',False)))
            if key not in self.slots:
                self.seen.discard(key)
        try:
            return super().velocity(original, **kwargs)
        finally:
            # A key evicted inside the parent must not remain a cache hit when
            # it returns later, including another branch in this generation.
            self.seen.intersection_update(self.slots)

    def _generate(self, original, *args, **kwargs):
        if self.request:
            raise RuntimeError('Reentrant Cosmos generation is unsupported')
        if (self.disabled or self.closed or torch.is_grad_enabled() or len(args)>1
                or kwargs.get('velocity_postprocess_builder') is not None
                or kwargs.get('upsample_task') is not None):
            self.invalidate()
            self.stats['bypasses']+=1
            return original(*args, **kwargs)
        if any(module.training for module in self.model.modules()):
            self.invalidate()
            self.stats['bypasses']+=1
            return original(*args, **kwargs)
        try:
            signature=(model_signature(self.model),repr(tuple(kwargs.get(name) for name in
                       ('guidance','num_steps','shift','guidance_interval','normalize_cfg'))))
        except BaseException:
            self.invalidate()
            self.persistent_stats['aborts']+=1
            raise
        if self.signature is not None and signature != self.signature:
            self.invalidate()
            self.persistent_stats['invalidations']+=1
        self.signature=signature
        self.request=True
        self.seen=set(self.published).intersection(self.slots)
        self.checks.clear()
        try:
            result=original(*args, **kwargs)
            if self.disabled:
                self.invalidate()
            else:
                self.published=set(self.seen).intersection(self.slots)
                self.persistent_stats['commits']+=1
            self.stats['requests']+=1
            return result
        except BaseException:
            self.invalidate()
            self.persistent_stats['aborts']+=1
            raise
        finally:
            self.request=False
            self.active=None
            self.filling=False
            self.validating=False
            self.seen.clear()
            self.checks.clear()


class PersistentConditioningCache(PersistentTransactions, ConditioningCache):
    def __init__(self, model, *, verify=False, max_slots=4):
        super().__init__(model,verify=verify,max_slots=max_slots)
        self.persistent_init()

    def report(self):
        return dict(super().report(),persistent=dict(self.persistent_stats),
                    published_slots=len(self.published),scope='fixed-geometry offline generation cache')

    def close(self):
        with self.lock:
            self.invalidate()
            super().close()
