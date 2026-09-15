"""Narrow startup-only compatibility fixes for the pinned Omni build on Thor."""
import hashlib,json
from pathlib import Path
root=Path('/home/guanming/ifl_eval/framework_comparison_20260913/omni-thor-compat-v1')
changes=[]
def edit(rel,old,new):
 p=root/rel;s=p.read_text();assert s.count(old)==1
 updated=s.replace(old,new);p.write_text(updated)
 changes.append({'path':rel,'before_sha256':hashlib.sha256(s.encode()).hexdigest(),'after_sha256':hashlib.sha256(updated.encode()).hexdigest(),'old':old,'new':new})
edit('vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py','prompt == "dummy run" and req.sampling_params.num_inference_steps == 1','prompt == "dummy run" and req.sampling_params.num_inference_steps in (1, 2)')
# Reclaim clean checkpoint cache before querying CUDA free memory on unified-memory Thor.
edit('vllm_omni/experimental/ar_diffusion/runner.py','return int(torch.cuda.mem_get_info(self.device)[0])','''import os
        from pathlib import Path
        for path in (Path.home() / ".cache/huggingface/hub").glob("models--*/blobs/*"):
            if path.is_file() and path.stat().st_size >= 64 * 1024 * 1024:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        return int(torch.cuda.mem_get_info(self.device)[0])''')
(root/'compatibility_manifest.json').write_text(json.dumps({'base_commit':'f7d9deb45ab56e6a2ccc1690279bd9e6bdefbfe3','scope':'startup only: accept the upstream two-step dummy warmup and reclaim clean checkpoint pages before memory admission','changes':changes},indent=2)+'\n')
