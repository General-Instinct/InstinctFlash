# Frozen Edge artifact storage

[Receipt](receipt.json) records completed Thor deduplication of identical safetensors in eight frozen V16/V17 packages. SHA256 and size were checked before atomic hardlink replacement, then all 40 weight paths and eight manifests were verified again. Unique weight inodes decreased from 40 to 10; free space rose from 41,828,667,392 to 97,043,697,664 bytes (55.22 GB recovered).

All paths and bytes remain available, including excluded preparations. Configs, manifests and producer outputs were preserved. The operation held the shared Thor GPU lock. These files are immutable artifacts: subsequent package creation must copy or replace paths rather than modify shared inodes in place. The receipt binds the exact script hash; this is storage maintenance, not performance evidence.
