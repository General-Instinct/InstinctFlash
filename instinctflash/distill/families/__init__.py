"""Built-in family adapters. Importing this package registers them.

Two families are instantiated (wan_va, pi05 — the two the campaign has measured end to end);
the other six supported serving families are sketched in the RFC and join by adding a module
here. Registration is import-time and idempotent per process, mirroring how serving passes
register as builtins.
"""

from instinctflash.distill.families import pi05, wan_va  # noqa: F401
