# Third-party notices

The FastDialogueSidon source code and model weights are licensed under the
Apache License 2.0 (`LICENSE`). The code contains parts derived from, or
vendored from, the MIT-licensed projects below; their copyright and
permission notices are reproduced here as the MIT License requires.

| Component | Where | Upstream | License |
|---|---|---|---|
| Model definitions adapted from the Sidon codebase | `trt/src/fastdialoguesidon/modeling.py`, `triton/src/fds_triton/reference/modeling.py` | [sarulab-speech/Sidon](https://github.com/sarulab-speech/Sidon) | MIT, Copyright (c) 2025 sarulab-speech |
| WaveFitNeXt generator and audio utilities (vendored) | `trt/src/fastdialoguesidon/wavefit_next.py`, `wavefit_audio_utils.py` (and the copies under `triton/src/fds_triton/reference/`) | [yukara-ikemiya/wavefit-pytorch](https://github.com/yukara-ikemiya/wavefit-pytorch) | MIT, Copyright (c) 2024 Yukara Ikemiya |

Runtime dependencies are installed separately and keep their own licenses:
PyTorch / torchaudio (BSD-style), Triton (MIT), Hugging Face Transformers
(Apache-2.0), safetensors (Apache-2.0), and — for the `trt/` backend only —
NVIDIA TensorRT, which is distributed under NVIDIA's proprietary license
agreement and is not included in this repository.

## Model weights

The FastDialogueSidon weights
([higurashi-ai/FastDialogueSidon](https://huggingface.co/higurashi-ai/FastDialogueSidon))
are released under the **Apache License 2.0**, like the code. The encoder
uses the w2v-BERT 2.0 architecture
([facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), MIT).

---

## MIT License — sarulab-speech/Sidon

```
MIT License

Copyright (c) 2025 sarulab-speech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIT License — yukara-ikemiya/wavefit-pytorch

```
MIT License

Copyright (c) 2024 Yukara Ikemiya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
