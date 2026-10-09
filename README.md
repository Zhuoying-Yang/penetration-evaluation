# Penetration Evaluation

Automated evaluation of object-environment penetration in AI-generated robot manipulation videos.

This repository contains the frozen **Type A penetration evaluation pipeline**, which detects visually implausible interactions where an object appears to pass through a solid environmental boundary rather than following a physically plausible trajectory.

## Overview

Video generation models can produce visually realistic robot manipulation sequences while violating basic physical constraints. One failure mode is object-environment penetration, such as a manipulated object passing through the solid wall of a container.

Our goal is to evaluate these failures automatically using visual evidence, without requiring ground-truth 3D geometry.

The current implementation focuses on **Type A: solid-boundary penetration**.

## Evaluation Pipeline

The pipeline consists of four main stages:

### 1. Agent 0: Interaction Selection

- Analyze the video to identify the manipulated object and relevant environment.
- Select candidate actor-target pairs.
- Identify the solid boundary involved in the interaction.
- Prioritize object-environment interactions.

This stage does not use human penetration labels.

### 2. Qwen: Frame-Level Penetration Detection

A vision-language model analyzes frame transitions throughout the video.

The evaluator looks for visual evidence suggesting that an object crosses a solid environmental boundary.

The original Type A evaluation prompt is preserved, with literal substitution of the selected actor and target names.

### 3. Temporal Consistency

Frame-level predictions are processed to identify meaningful, temporally consistent penetration candidates.

This reduces the influence of isolated observations and selects candidate frames for further verification.

### 4. Gemini: Candidate Verification

A second vision-language model independently examines selected candidates using full-frame and local-region evidence.

The verification stage considers:

- Whether the correct manipulated object is identified.
- Whether the observed interaction involves a solid boundary.
- Whether ordinary occlusion or a legitimate opening can explain the appearance.
- Whether the visual evidence supports a penetration violation.

The final video-level score is aggregated from the verified candidates.

## Repository Structure

| File | Description |
|------|-------------|
| `penetration_typeA_agent0_literal_batch.py` | Main end-to-end Type A evaluation pipeline |
| `penetration_agent0_v3A_literal.py` | Agent 0 interaction selection and prompt configuration |
| `penetration_abc_agents_v3.py` | Qwen frame-level evaluation and temporal processing |
| `penetration_A_v3_crop_gemini_score.py` | Gemini verification and scoring utilities |
| `penetration_typeA_manifest.csv` | Evaluation video list and human annotations |

## Preliminary Results

The current pipeline was evaluated on a small curated benchmark containing **13 robot manipulation videos**, including 3 positive and 10 negative examples.

| Evaluation Stage | AUROC |
|------------------|-------|
| Raw Qwen predictions | 0.3833 |
| Qwen with temporal postprocessing | 0.6167 |
| Full pipeline with Gemini verification | **0.8167** |

For the container-wall subset (2 positives and 10 negatives), the full pipeline achieved an AUROC of **1.0000**.

These results are preliminary because the evaluation dataset is small and was used during method development. Performance on unseen videos remains to be established.

## Usage

The scripts were developed and evaluated in a configured HPC research environment with local Qwen inference and a Gemini backend.

Input videos, model weights, cloud credentials, and the external laboratory VLM runtime are not distributed in this repository.

### Configure the Environment

Set the working directory and Python environment for your deployment.

For the original ERIS setup:

    ROOT=/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1
    QPY=/PHShome/zy992/Wilson/dependency/env/qwen/bin/python

    export PYTHONPATH="$ROOT/.v4_python_deps${PYTHONPATH:+:$PYTHONPATH}"

The evaluation manifest contains absolute video paths specific to the original environment. Update them when using different storage locations.

### Check the Dataset

    "$QPY" "$ROOT/penetration_typeA_agent0_literal_batch.py" check

### Evaluate a Single Video

    "$QPY" -u "$ROOT/penetration_typeA_agent0_literal_batch.py" run-one \
      --case COSMOS25_0003

### Run the Full Benchmark

    "$QPY" -u "$ROOT/penetration_typeA_agent0_literal_batch.py" batch

### Generate the Evaluation Report

    "$QPY" "$ROOT/penetration_typeA_agent0_literal_batch.py" report

The pipeline supports cached intermediate results to facilitate repeated evaluation and analysis.

## Limitations

- This approach evaluates visual evidence of penetration, not exact 3D geometric intersection.
- Normal occlusion and legal insertion through container openings can be difficult to distinguish from penetration.
- Model-based verification can produce both false positives and false negatives.
- Results depend on object selection, temporal interpretation, and the reliability of the VLMs.
- The current benchmark is small and does not establish generalization to unseen video generators or manipulation tasks.

## Future Work

Future extensions may include:

- Object persistence and geometric consistency.
- More reliable separation of penetration and ordinary occlusion.
- Depth and point-tracking-based verification.
- Evaluation across larger and more diverse robot manipulation benchmarks.

## Status

**Type A: Frozen research implementation**

Other penetration categories and experimental extensions remain under development and are not included in this release.
