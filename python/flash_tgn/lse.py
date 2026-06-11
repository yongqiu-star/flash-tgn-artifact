from __future__ import annotations

import torch
from torch import Tensor, nn

from .extension import cuda_extension
from .layers import TimeEncode


class LSEEngine(nn.Module):
    """Local State Engine for GRU-based memory updates."""

    def __init__(self, d_mem: int, d_mail: int, d_time: int):
        super().__init__()
        self.d_mem = d_mem
        self.d_time = d_time
        self.mem_time_encode = TimeEncode(d_time)
        self.gru = nn.GRUCell(d_mail + d_time, d_mem)

    def forward(
        self,
        nodes: Tensor,
        mailbox_data: Tensor,
        mailbox_ts: Tensor,
        mem_data: Tensor,
        mem_ts: Tensor,
        device,
    ):
        """Returns `(new_mem, new_ts)` for `nodes` on `device`."""
        nodes_l = nodes.long().to(mailbox_data.device)
        try:
            _C = cuda_extension()
            mail, mail_ts, old_mem, old_ts = _C.fused_lse_gather(
                mailbox_data,
                mailbox_ts.squeeze(),
                mem_data,
                mem_ts.squeeze(),
                nodes_l,
            )
        except Exception:
            mail = mailbox_data[nodes_l]
            mail_ts = mailbox_ts[nodes_l]
            old_mem = mem_data[nodes_l]
            old_ts = mem_ts[nodes_l]

        if mail_ts.dim() == 2:
            mail_ts = mail_ts.squeeze(-1)
        if old_ts.dim() == 2:
            old_ts = old_ts.squeeze(-1)
        delta = mail_ts - old_ts
        t_enc = self.mem_time_encode(delta)
        mail_full = torch.cat([mail, t_enc], dim=1)
        new_mem = self.gru(mail_full, old_mem)
        return new_mem, mail_ts
