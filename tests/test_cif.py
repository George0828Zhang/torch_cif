import unittest
import torch
from typing import Optional
from torch import Tensor
from torch_cif import cif_function

import hypothesis.strategies as st
from hypothesis import assume, given, settings
from torch.testing._internal.common_utils import TestCase

TEST_CUDA = torch.cuda.is_available()


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class TestCIF(TestCase):
    def _test_cif_ref(
        self,
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

        for b in range(B):
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

    def _test_custom_cif_impl(
        self, *args, **kwargs
    ):
        return cif_function(*args, **kwargs)

    @settings(deadline=None)
    @given(
        B=st.integers(1, 10),
        T=st.integers(1, 20),
        S=st.integers(1, 200),
        C=st.integers(1, 20),
        beta=st.floats(0.5, 1.5),
        device=st.sampled_from(["cpu", "cuda"]),
    )
    def test_cif(self, B, T, S, C, beta, device):

        assume(device == "cpu" or TEST_CUDA)

        # inputs
        device = torch.device("cpu")
        # inputs
        input = torch.rand(B, S, C, device=device)
        alpha = torch.randn((B, S), device=device).sigmoid_()
        # source_lengths = torch.full((B,), S, device=device)
        # target_lengths = torch.full((B,), T, device=device)
        source_lengths = torch.randint(1, S + 1, (B,), device=device)
        target_lengths = torch.randint(1, T + 1, (B,), device=device)

        source_lengths = (source_lengths * S / source_lengths.max()).long()
        target_lengths = (target_lengths * T / target_lengths.max()).long()

        padding_mask = lengths_to_padding_mask(source_lengths)

        # train
        y, dy = self._test_cif_ref(
            input,
            alpha,
            beta,
            padding_mask=padding_mask,
            target_lengths=target_lengths
        )

        x_out = self._test_custom_cif_impl(
            input,
            alpha,
            beta,
            padding_mask=padding_mask,
            target_lengths=target_lengths
        )
        x = x_out['cif_out'][0]
        dx = x_out['delays'][0]
        torch.testing.assert_close(
            x,
            y,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            dx,
            dy,
            atol=1e-3,
            rtol=1e-3,
        )

        # test
        y2, dy2 = self._test_cif_ref(
            input,
            alpha,
            beta,
            padding_mask=padding_mask
        )

        x2_out = self._test_custom_cif_impl(
            input,
            alpha,
            beta,
            padding_mask=padding_mask
        )
        x2 = x2_out['cif_out'][0]
        dx2 = x2_out['delays'][0]
        torch.testing.assert_close(
            x2,
            y2,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            dx2,
            dy2,
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
