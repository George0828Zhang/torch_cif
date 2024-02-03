# torch-cif

A fast parallel implementation pure PyTorch implementation of *"CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition"*  https://arxiv.org/abs/1905.11235.

## Installation
### PYPI
```bash
pip install torch-cif
```
### Locally
```bash
git clone https://github.com/George0828Zhang/torch_cif
cd torch_cif
python setup.py install
```

## Usage
```python
def cif_function(
    inputs: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
    unbound_alpha: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235

    Shapes:
        N: batch size
        S: source (encoder) sequence length
        C: source feature dimension
        T: target sequence length

    Args:
        inputs (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            inputs. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the inputs.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4
        unbound_alpha (bool, optional): Whether to check if 0 <= alpha <= 1.

    Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
        cif_out (Tensor): (N, T, C) The output integrated from the source.
        cif_lengths (Tensor): (N,) The output length for each element in batch.
        alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays (Tensor): (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights (Tensor, optional): (N,) During inference, return the tail.
    """
```

## Note
- This implementation uses `cumsum` and `floor` to determine the firing positions, and use `scatter` to merge the weighted source features. The figure below demonstrates this concept using *scaled* weight sequence `(0.4, 1.8, 1.2, 1.2, 1.4)`

<img src="concept.png" alt="drawing" width="300"/>

- Runing test requires `pip install hypothesis expecttest`.
- If `beta != 1`, our implementation slightly differ from Algorithm 1 in the paper [[1]](#reference):
    - When a boundary is located, the original algorithm add the last feature to the current integration with weight `1 - accumulation` (line 11 in Algorithm 1), which causes negative weights in next integration when `alpha < 1 - accumulation`. 
    - We use `beta - accumulation`, which means the weight in next integration `alpha - (beta - accumulation)` is always positive.
- Feel free to contact me if there are bugs in the code.

## References
1. [CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition](https://arxiv.org/abs/1905.11235)
2. [Exploring Continuous Integrate-and-Fire for Adaptive Simultaneous Speech Translation](https://www.isca-archive.org/interspeech_2022/chang22f_interspeech.html)