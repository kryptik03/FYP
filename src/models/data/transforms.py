"""
transforms.py
=============
Signal pre-processing transforms for the FYP PD pipeline.

Currently provides:
    DecimateMaxPool1D - shrinks a long 1-D waveform via max-pooling so that
                        transient PD spike peaks are preserved.
"""

import torch
import torch.nn.functional as F


class DecimateMaxPool1D:
    """
    Decimates a 1-D signal using 1-D max-pooling.

    Why max-pooling instead of average pooling or plain sub-sampling?
    ----------------------------------------------------------------
    PD pulses are very short transients (nanoseconds) sitting inside a 5 us
    window.  Average pooling would dilute the spike amplitude across 500
    neighbouring samples, making it hard for the CNN to see.  Max-pooling
    instead keeps the single highest absolute value inside each window, which
    is exactly the peak of the spike - the most diagnostic feature.

    Input shape:  (C, N)   - channel-first, no batch dimension
    Output shape: (C, N // factor)

    Example
    -------
    >>> dec = DecimateMaxPool1D(factor=500)
    >>> x   = torch.randn(1, 500001)   # single-channel raw signal
    >>> y   = dec(x)
    >>> y.shape
    torch.Size([1, 1000])
    """

    def __init__(self, factor: int):
        """
        Args:
            factor: How many raw samples collapse into one decimated sample.
                    E.g., factor=500 turns 500 001 points into 1 000 points.
        """
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (C, N).

        Returns:
            Decimated float tensor of shape (C, floor((N - factor) / factor) + 1).
            For N=500 001, factor=500 this is exactly (C, 1000).
        """
        # F.max_pool1d requires a batch dimension: (B, C, N)
        x = x.unsqueeze(0)                                      # (1, C, N)
        x = F.max_pool1d(x, kernel_size=self.factor,
                         stride=self.factor)                     # (1, C, N_dec)
        x = x.squeeze(0)                                        # (C, N_dec)
        return x
