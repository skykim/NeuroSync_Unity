# NeuroSync for Unity

[![Unity Version](https://img.shields.io/badge/Unity-6000.0.50f1+-black.svg?style=for-the-badge&logo=unity)](https://unity.com/)
[![Inference Engine](https://img.shields.io/badge/Inference-2.2.1-blue.svg?style=for-the-badge)](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.2/manual/index.html)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/AnimaVR/NEUROSYNC)

This project allows you to use the NeuroSync lipsync model directly in Unity projects via the **Unity AI Inference Engine** (formerly Sentis).

## ✨ Key Features

- **On-Device Inference**: Runs the NeuroSync model directly on the user's device using the Inference Engine.
- **Local API Server**: Includes an example of a Flask-based API server for model inference as an alternative.

## ⚙️ Requirements

- **Unity**: `6000.0.50f1`
- **Inference Engine**: `2.2.1`

## 🧠 Model (ONNX)

This project requires the NeuroSync model in the ONNX format.

1.  The original model is available at the [AnimaVR Hugging Face repository](https://huggingface.co/AnimaVR/NEUROSYNC).
2.  You can convert the original model to ONNX format using the resources available at [skykim/NeuroSync_Local_API_ONNX](https://github.com/skykim/NeuroSync_Local_API_ONNX). Please follow the instructions in that repository to generate the `.onnx` file.

### Parity with the Python reference (11/11/2026)

The on-device path now matches the Python NeuroSync pipeline to float32 precision
(final blendshapes within ~4e-5 on CPU and GPUCompute, Inference Engine 2.2.1 and 2.6.1):

- Re-export `audio_feature_extractor.onnx` with the updated exporter in
  [NeuroSync_Local_API_ONNX](https://github.com/skykim/NeuroSync_Local_API_ONNX). Models made with the old exporter differ from librosa by up to ~10 per feature.
- Select the ONNX files and run **Tools ▸ NeuroSync ▸ Serialize Selected ONNX to StreamingAssets**
  (writes `<name>.sentis` fp32 and `<name>_fp16.sentis`). The controller loads
  `audio_feature_extractor.sentis` (keep it fp32) and `neurosync_fp16.sentis` by default; both names are fields on `NeurosuncController`.
- `AudioUtils` resamples with a Kaiser-windowed sinc filter and normalises after resampling, as librosa does.
  The old linear interpolation changed the final blendshapes by up to ~4.6.

## 🚀 Getting Started

### 1. Project Setup

1.  Clone or download this repository.
2.  Download and unzip the provided **[StreamingAssets.zip](https://drive.google.com/file/d/1ruLW-G1EcVpA80muLV_7OLDIaTXzemBB/view?usp=sharing)**.
3.  Place its contents into the `/Assets/StreamingAssets` directory in your Unity project.

### 2. Run the Demo Scene

1.  Open the `/Assets/Scenes/LipSyncScene.unity` scene in the Unity Editor.
2.  Enter Play Mode.
3.  Press the `SpaceBar` key to play the demo audio and see the lipsync in action.

## 🧪 Test Prompts

The demo scene cycles through the following pre-recorded audio prompts when you press the spacebar:

> 1.  (English) As a good friend once said, manners maketh man.
> 2.  (German) Wie mein guter Freund immer sagt: Gute Manieren machen erst den Menschen aus.
> 3.  (Japanese) いい友達が言ってたんだけど、礼儀正しさが人を作るんだって。
> 4.  (Korean) 좋은 친구가 그러더라고, 예의가 사람을 만든다고.

[![NeuroSync in Unity](https://img.youtube.com/vi/w8UOyJVR37A/0.jpg)](https://www.youtube.com/watch?v=w8UOyJVR37A)
