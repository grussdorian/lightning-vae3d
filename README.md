# lightning-vae3d
ResNet-like variational autoencoder (VAE) for encoding 3D volume elements described in the paper "3D variational
autoencoder for fingerprinting microstructure volume elements", which can be found
[here](https://arxiv.org/abs/2503.17427).
All supporting data, including microstructure volume elements and pretrained model checkpoints can be found on
[Zenodo](https://zenodo.org/records/15261939).


## Installation


### Install with `pip`

```
git clone https://github.com/micmog/lightning-vae3d
cd lightning-vae3d
pip install .
```


## Examples

Example scripts for training the VAE are provided and only require modification on the `TRAIN_DIR` and `VAL_DIR`
parameters to point to the relevant data. Pretrained models can be loaded by setting the parameter `load_model=True` and
specifying a `checkpoint_path`.

An example of a simple fully connected network that was used as the surrogate model in the
[paper](https://arxiv.org/abs/2503.17427) is also available.


## License (BSD 3-Clause)

Copyright (c) 2025, United Kingdom Atomic Energy Authority

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
