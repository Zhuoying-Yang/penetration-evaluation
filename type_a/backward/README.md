# Type A Backward / Reverse — experimental freeze

**Qwen full-video candidate detector** (ERIS source variants) followed by an independent **trajectory-validity verifier** (Sol macOS scripts). This combination has not yet been run and evaluated as a unified GT-blind pipeline.

- `eris/penetration_typeA_reverse_v1.py` and `eris/penetration_typeA_reverse_v1_1.py`: reverse candidate stage variants.
- `eris/typeA_reverse_temporal_v4.py`: temporal reverse diagnostic.
- `eris/reverse_transparent_vlm_verifier_v1.py`, `eris/reverse_transparent_vlm_v2.py`: earlier local Qwen verifiers; not the current preferred result.
- `eris/typeA_reverse_gemini_*`: Gemini diagnostic variants, not the frozen Forward baseline.
- `mac/sol_transparent_moreframes_v2.py`: prepare ordered frames and call gateway.
- `mac/sol_transparent_multiscale_v1.py`: add fixed close-up crops.
- `mac/transparent_trajectory_prompt_v3.py`: **latest trajectory-validity prompt**; run before the gateway judge script.
- `diagnostics/`: edge tracking and geometry explorations (known to drift on transparent material).

The tested windows for 0055, 0056, 0057 were picked manually. An automated end-to-end video-level metric is future work. Do not train or evaluate on these three development cases as held-out evidence.
