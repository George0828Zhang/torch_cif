import torch
from typing import Optional
from torch import Tensor

import torch.utils.benchmark as benchmark


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def cif_sequential_ref(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
) -> Tensor:
    B, S, C = input.size()

    if padding_mask is not None:
        alpha = alpha.masked_fill(padding_mask, 0)

    if target_lengths is not None:
        feat_lengths = target_lengths.long()
        desired_sum = beta * target_lengths.type_as(input) + eps
        alpha_sum = alpha.sum(1)
        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        T = feat_lengths.max()
    else:
        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

    output = input.new_zeros((B, T + 1, C))
    delay = input.new_zeros((B, T + 1))

    if padding_mask is not None:
        source_lengths = (~padding_mask).sum(-1).long()
    else:
        source_lengths = input.new_full((B,), S, dtype=torch.long)

    # for b in range(B):
    assert B == 1
    b = 0

    csum = 0
    src_idx = 0
    dst_idx = 0
    tail_idx = 0
    while src_idx < source_lengths[b]:
        if csum + alpha[b, src_idx] < beta:
            csum += alpha[b, src_idx]
            output[b, dst_idx] += alpha[b, src_idx] * input[b, src_idx]
            delay[b, dst_idx] += alpha[b, src_idx] * (1 + src_idx) / beta
            tail_idx = dst_idx
            alpha[b, src_idx] = 0
            src_idx += 1
        else:
            fire_w = beta - csum
            alpha[b, src_idx] -= fire_w
            output[b, dst_idx] += fire_w * input[b, src_idx]
            delay[b, dst_idx] += fire_w * (1 + src_idx) / beta
            tail_idx = dst_idx
            csum = 0
            dst_idx += 1

    if csum >= tail_thres:
        output[b, tail_idx] *= beta / csum
    else:
        output[b, tail_idx:] = 0

    # tail handling
    if (target_lengths is not None) or output[:, T, :].eq(0).all():
        # training time -> ignore tail
        output = output[:, :T, :]
        delay = delay[:, :T]

    return output, delay


if __name__ == "__main__":
    # B, S, T, C = 256, 3072, 512, 256
    B, S, T, C = 1, 1024, 512, 256
    beta = 0.5

    # inputs
    device = torch.device("cuda:0")
    # inputs
    input = torch.rand(B, S, C, device=device)
    alpha = torch.randn((B, S), device=device).sigmoid_()
    source_lengths = torch.full((B,), S, device=device)
    target_lengths = torch.full((B,), T, device=device)

    padding_mask = lengths_to_padding_mask(source_lengths)
    
    globals = {
        'input': input,
        'alpha': alpha,
        'beta': beta,
        'padding_mask': padding_mask,
        'target_lengths': target_lengths,
    }

    num_threads = torch.get_num_threads()
    print(f'Benchmarking on {num_threads} threads')

    t1 = benchmark.Timer(
        stmt='cif_function(input,alpha,beta,padding_mask=padding_mask,target_lengths=target_lengths)',
        setup='from torch_cif import cif_function',
        globals=globals,
        num_threads=num_threads
    )

    t0 = benchmark.Timer(
        stmt='cif_sequential_ref(input,alpha,beta,padding_mask=padding_mask,target_lengths=target_lengths)',
        setup='from __main__ import cif_sequential_ref',
        globals=globals,
        num_threads=num_threads
    )

    print(t1.timeit(1000))
    print(t0.timeit(1000))
