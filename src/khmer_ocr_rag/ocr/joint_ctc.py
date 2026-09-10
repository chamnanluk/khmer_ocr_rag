from __future__ import annotations

"""Research implementation of the proposal's joint Khmer CTC architecture.

The implementation follows the *architectural idea* described in the reviewed KTRWS
paper: visual encoder → modality-aware feature selector → FiLM conditioning from a
binary boundary-output flag → CTC head whose vocabulary includes U+200B.

It is intentionally not labeled an exact reproduction of external/review-gated code.
"""


class JointCTCRecognizer:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        layers: int = 3,
        n_adapters: int = 5,
        dropout: float = 0.1,
    ):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("Install OCR extras: pip install -e '.[ocr]'") from exc
        self.torch = torch
        self.nn = nn

        class ResidualBlock(nn.Module):
            def __init__(self, cin, cout, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(cout)
                self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(cout)
                self.act = nn.ReLU(inplace=True)
                self.skip = (
                    nn.Identity() if cin == cout and stride == 1
                    else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
                )

            def forward(self, x):
                y = self.act(self.bn1(self.conv1(x)))
                y = self.bn2(self.conv2(y))
                return self.act(y + self.skip(x))

        class MAFS(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(d_model, n_adapters)
                self.adapters = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_model // 4), nn.ReLU(),
                        nn.Linear(d_model // 4, d_model),
                    ) for _ in range(n_adapters)
                ])

            def forward(self, x):
                # x: B,T,D. Global pooling selects a soft mixture of learned adapters.
                z = x.mean(dim=1)
                r = self.router(z).softmax(dim=-1)  # B,N
                adapted = torch.stack([a(x) for a in self.adapters], dim=-2)  # B,T,N,D
                mixed = (adapted * r[:, None, :, None]).sum(dim=-2)
                return x + mixed, r

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    ResidualBlock(3, 32, stride=2),
                    ResidualBlock(32, 64),
                    ResidualBlock(64, 128),
                    ResidualBlock(128, 256, stride=2),
                    ResidualBlock(256, 384),
                    ResidualBlock(384, d_model),
                )
                enc = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
                self.mafs = MAFS()
                self.boundary_embed = nn.Embedding(2, d_model)
                self.boundary_proj = nn.Linear(d_model, d_model)
                self.gamma = nn.Linear(d_model, d_model)
                self.beta = nn.Linear(d_model, d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, images, boundary_flag):
                feat2d = self.cnn(images)                       # B,D,H',W'
                seq = feat2d.mean(dim=2).transpose(1, 2)        # B,W',D
                seq = self.encoder(seq)
                seq, routing = self.mafs(seq)
                eb = self.boundary_proj(self.boundary_embed(boundary_flag.long()))
                gamma = self.gamma(eb).unsqueeze(1)
                beta = self.beta(eb).unsqueeze(1)
                conditioned = gamma * seq + beta
                return self.head(conditioned), routing          # B,T,V ; B,N

        self.model = Model()

    def ctc_loss(self, logits, targets, input_lengths, target_lengths, blank_id: int = 0):
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        return self.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_id,
            zero_infinity=True,
        )

    @staticmethod
    def output_lengths_from_pixel_widths(pixel_widths):
        # Two stride-2 residual blocks reduce width by ~4. Convolution padding makes ceil
        # behavior model-dependent; this expression matches Conv2d k=3,p=1,s=2 twice.
        return ((pixel_widths + 1) // 2 + 1) // 2
