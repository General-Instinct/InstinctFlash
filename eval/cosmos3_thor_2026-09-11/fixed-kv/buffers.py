"""Experimental fixed KV storage preserving Cosmos's interleaved token order."""
import torch
from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq


def install(service):
    cache = service._ifl_conditioning_cache
    stats = dict(prefills=0, updates=0, bypasses=0)
    for index, layer in enumerate(service.model.net.language_model.model.layers):
        attention = layer.self_attn
        original = attention.dispatch_attention_fn
        def dispatch(q, k, v, mask, *, _original=original, _index=index, **kwargs):
            if cache.active is None or cache.disabled or cache.validating:
                stats['bypasses'] += 1
                return _original(q, k, v, mask, **kwargs)
            normalized = kwargs.get('packed_key_states_normalized')
            key_pack = normalized if normalized is not None else k
            def fixed(pack, name):
                if 'all_seq' in pack:
                    return pack
                assert not pack['is_sharded']
                buffers = cache.active.setdefault('kv_buffers', {})
                key = (_index, name)
                if cache.filling:
                    reference = get_all_seq(pack)
                    if key not in buffers:
                        buffers[key] = reference.clone()
                    else:
                        assert buffers[key].shape == reference.shape
                        buffers[key].copy_(reference)
                    stats['prefills'] += 1
                    # Feed the exact native merge to attention, avoiding a
                    # second identical merge while retaining independent storage.
                    result = dict(pack)
                    result['all_seq'] = reference
                    return result
                buffer = buffers[key]
                full_indices = pack['_full_indices']
                # Same scatter and index order as native get_all_seq. The fixed
                # text rows were populated during this request's branch prefill.
                buffer[full_indices] = pack['full_only_seq'][:full_indices.shape[0]]
                result = dict(pack)
                result['all_seq'] = buffer
                stats['updates'] += 1
                return result
            k_fixed, v_fixed = fixed(key_pack, 'k'), fixed(v, 'v')
            if normalized is not None:
                kwargs['packed_key_states_normalized'] = k_fixed
            else:
                k = k_fixed
            return _original(q, k, v_fixed, mask, **kwargs)
        cache.patch(attention, 'dispatch_attention_fn', dispatch)
    return stats
