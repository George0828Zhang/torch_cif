# torch-cif

A fast parallel implementation pure PyTorch implementation of *"CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition"*  https://arxiv.org/abs/1905.11235.

## Usage
```python
def cif_function(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235

    Args:
        input (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            input. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the input.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4

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

- Run test by `python test.py` (requires `pip install expecttest`).
- If `beta != 1`, our implementation slightly differ from Algorithm 1 in the paper [[1]](#reference):
    - When a boundary is located, the original algorithm add the last feature to the current integration with weight `1 - accumulation` (line 11 in Algorithm 1), which causes negative weights in next integration when `alpha < 1 - accumulation`. 
    - We use `beta - accumulation`, which means the weight in next integration `alpha - (beta - accumulation)` is always positive.
- Feel free to contact me if there are bugs in the code.

## Reference
1. [CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition](https://arxiv.org/abs/1905.11235)