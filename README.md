# torch-cif

A pure PyTorch batched computation implementation of *"CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition"*  https://arxiv.org/abs/1905.11235.

## Usage
```python
def cif_function(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    max_output_length: Optional[int] = None,
    eps: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor]:
    r""" A batched computation implementation of continuous integrate and fire (CIF)
    https://arxiv.org/abs/1905.11235

    Args:
        input (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            input. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the input.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        max_output_length (int, optional): The maximum valid output length used
            in inference. The alpha is scaled down if the sum exceeds this value.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4

    Returns: Tuple (output, feat_lengths, alpha_sum)
        output (Tensor): (N, T, C) The output integrated from the source.
        feat_lengths (Tensor): (N,) The output length for each element in batch.
        alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
    """
```

## Note
> :information_source: This is a WIP project. the implementation is still being tested.
- This implementation uses `cumsum` and `floor` to determine the firing positions, and use `scatter` to merge the weighted source features.
- Run test by `python test.py` (requires `pip install expecttest`).
- Feel free to contact me if there are bugs in the code.

## Reference
- [CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition](https://arxiv.org/abs/1905.11235)