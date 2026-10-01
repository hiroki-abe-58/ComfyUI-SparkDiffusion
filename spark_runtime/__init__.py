"""Host-side integration of the isolated SparkDiffusion runtime.

Importing this package is cheap: it never imports torch, triton, flash-attn or
the SparkDiffusion repository. Heavy work happens in a separate interpreter via
``launcher.py``.
"""
