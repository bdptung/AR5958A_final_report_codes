import json
import torch
import torch.nn as nn
import numpy as np

def make_embedding(num_tokens: int, emb_dim: int, pad_idx: int = 0, zero_pad: bool = True) -> nn.Embedding:
    emb = nn.Embedding(num_tokens, emb_dim, padding_idx=pad_idx)
    if zero_pad:
        with torch.no_grad():
            emb.weight[pad_idx].zero_()  # background token contributes nothing
    return emb

def save_embedding(emb: nn.Embedding, path: str, extra: dict = None):
    """Saves just what you need: weights + essential metadata."""
    payload = {
        "state_dict": emb.state_dict(),
        "num_embeddings": emb.num_embeddings,
        "embedding_dim": emb.embedding_dim,
        "padding_idx": emb.padding_idx,
    }
    if extra:
        payload["extra"] = extra  # e.g., your string→token vocab mapping
    torch.save(payload, path)

def load_embedding(path: str, device: str = "cpu") -> nn.Embedding:
    payload = torch.load(path, map_location=device)
    emb = nn.Embedding(
        num_embeddings=payload["num_embeddings"],
        embedding_dim=payload["embedding_dim"],
        padding_idx=payload.get("padding_idx", None),
    )
    emb.load_state_dict(payload["state_dict"])
    return emb.to(device)

def embed_token_raster(tokens, emb: nn.Embedding, device: str = "cpu") -> torch.Tensor:
    """
    tokens: numpy array or torch tensor of shape [H, W] or [1, H, W], dtype=int
    returns: torch tensor [E, H, W] where E = emb.embedding_dim
    """
    if isinstance(tokens, np.ndarray):
        t = torch.from_numpy(tokens)
    else:
        t = tokens

    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)        # [H, W]
    assert t.dim() == 2, "Expect [H, W] or [1, H, W]"

    t = t.long().to(device)     # token indices
    emb = emb.to(device).eval()

    emap = emb(t)               # [H, W, E]
    emap = emap.permute(2, 0, 1).contiguous()  # [E, H, W]
    return emap

# from util import read_raster, write_raster
# zones_tokens_data, transform, crs, nodata = read_raster("./data/zones_tokens.tif")

# def load_vocab(path_json):
#     with open(path_json, "r", encoding="utf-8") as f:
#         obj = json.load(f)
#     return obj["vocab"], obj["meta"]

# vocab, meta = load_vocab("afs.bzo_zone_v.vocab.json")
# emb = make_embedding(len(vocab) + 1, emb_dim=8, pad_idx=0)
# emb_map = embed_token_raster(zones_tokens_data, emb).detach().numpy()          # [8, 512, 512]
# save_embedding(emb, "afs.bzo_zone_v.emb.pt", extra={"vocab": vocab})
# write_raster(emb_map, './data/zones_tokens.tokenized.tif', transform=transform, crs=crs, nodata=nodata)