"""Advise Linux to reclaim clean checkpoint file pages; never changes file contents."""
import json, os
from pathlib import Path
root=Path.home()/'.cache/huggingface/hub'
count=0;size=0
for path in root.glob('models--*/blobs/*'):
    if not path.is_file() or path.stat().st_size < 64*1024*1024:
        continue
    fd=os.open(path,os.O_RDONLY)
    try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)
    count+=1;size+=path.stat().st_size
print(json.dumps({'files_advised':count,'bytes_advised':size,'operation':'POSIX_FADV_DONTNEED','contents_modified':False}))
