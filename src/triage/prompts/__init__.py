"""Single-sourced prompt content for every surface (KTD7, R8).

``fragments`` holds the category/queue label definitions, per-step
instructions, and prompt-assembly helpers. Both the pipeline steps and the
(later) single-prompt baseline assemble their prompts from these same pieces,
so the comparison stays equal-information (R8).
"""

from __future__ import annotations
