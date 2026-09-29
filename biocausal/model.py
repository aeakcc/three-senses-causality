"""ATLOP-style document-level relation extraction model (DESIGN.md §2).

* typed entity markers (8 special tokens, embeddings initialised from the type word)
* exact overlapping-window encoding for inputs longer than ``max_len`` (stride ``stride``),
  hidden states and last-layer attentions averaged over the windows that cover a token
* entity embedding = logsumexp over mention start-marker states
* localized context pooling from last-layer attentions (mean over heads and mentions)
* bilinear group head (block size ``block``), (s,o)/(o,s) logits averaged (undirected relations)
* cross-entropy over 9 classes with an optional weight on ``None``, or ``loss="atl"``: ATLOP's adaptive-thresholding
  loss for one label per pair (``None`` = threshold class, argmax decision unchanged)
* wave 2 (preregistration §5): set / soft-T marginal likelihood through ``label_masks``, per-corpus None classes
  (``n_extra_none``, BioREx c1) and corpus tag tokens (``[CORPUS_x]`` before the marked text)
* wave 3 (preregistration §7): ``pair_matrix_layers`` / ``pair_matrix_dim`` (entity-by-entity pair map, ``PairMatrix``),
  ``nv_weight`` (novelty head over the same pair representation) and ``kd_loss`` (ensemble distillation, biocausal.distill);
  all three are off by default and the default code path is unchanged (tests/test_w3_model.py freezes it)
* paper 2: ``aux_heads`` (data.AUX_CLASSES: pc, ev), one more linear head each over the same bilinear pair
  representation as ``nv_head``, CE with ignore_index -100 weighted ``aux_weight``; off by default

The encoder must return attentions: load it with ``attn_implementation="eager"`` (sdpa returns
none in transformers 5.x); ``_run`` raises if they are missing.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from biocausal.data import (AUX_CLASSES, LABEL2ID, LABELS, MARKER_INIT_WORD, MARKER_TOKENS, MARKERS, NONE_SLOTS,
                            NOVELTY, TYPE_ORDER, Doc, label_logw, mark_text, unpairable_gold)

Feature = dict[str, Any]


def corpus_token(corpus: str | None) -> str:
    """Corpus tag token; BioRED and BC8 (corpus None) share ``[CORPUS_biored]``."""
    return f"[CORPUS_{corpus or 'biored'}]"


def add_corpus_tokens(tokenizer: PreTrainedTokenizerBase, encoder: PreTrainedModel, corpora: list[str]) -> None:
    """One special token per corpus plus ``[CORPUS_biored]``; new embeddings get the backbone's default init."""
    tokenizer.add_tokens([corpus_token(c) for c in (None, *corpora)], special_tokens=True)
    encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def add_markers(tokenizer: PreTrainedTokenizerBase, encoder: PreTrainedModel) -> None:
    """Add the 8 marker tokens, resize embeddings, init each marker from its type word (idempotent)."""
    added = tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if not added:
        return
    emb = encoder.get_input_embeddings().weight
    with torch.no_grad():
        for t in TYPE_ORDER:
            word_ids = tokenizer(MARKER_INIT_WORD[t], add_special_tokens=False)["input_ids"]
            v = emb[word_ids].mean(0)
            for m in MARKERS[t]:
                emb[tokenizer.convert_tokens_to_ids(m)] = v


def featurize(doc: Doc, tokenizer: PreTrainedTokenizerBase, corpus: str | None = None, label_handling: str | None = None,
              hard_map: bool = False, corpus_tag: bool = False) -> Feature:
    """Tokenize the marked document; map every mention to the token index of its start marker.

    ``unpairable`` = types of gold relations with no candidate pair (counted as false negatives by evaluate).
    ``label_handling`` (training only, ``corpus`` = external corpus or None for BioRED/BC8): ``none_per_corpus`` turns
    the corpus's None labels into class 9 + NONE_SLOTS.index(corpus); every other mode adds ``label_logw``
    (data.label_logw) and removes the pairs it drops. ``corpus_tag``: ``[CORPUS_x] `` prefixes the marked text.
    ``novelty``: NOVELTY index of the gold relation's novelty flag, -100 (ignored) for None pairs and corpora without one.
    ``pc`` / ``ev`` (every AUX_CLASSES key): the pair's own aux label, -100 when it has none.
    """
    marked, mpos = mark_text(doc)
    if corpus_tag:
        tag = corpus_token(corpus)
        if tokenizer.convert_tokens_to_ids(tag) == tokenizer.unk_token_id:
            raise ValueError(f"{tag} is not in the tokenizer (model.add_corpus_tokens)")
        marked, mpos = f"{tag} {marked}", {i: p + len(tag) + 1 for i, p in mpos.items()}
    enc = tokenizer(marked, return_offsets_mapping=True, add_special_tokens=True, truncation=False, verbose=False)
    marker_ids = set(tokenizer.convert_tokens_to_ids(MARKER_TOKENS))
    start2tok = {off[0]: i for i, (tid, off) in enumerate(zip(enc["input_ids"], enc["offset_mapping"]))
                 if tid in marker_ids}
    ents = doc["entities"]
    order = list(ents)
    index = {e: i for i, e in enumerate(order)}
    logw = label_logw(doc, corpus, label_handling, hard_map) \
        if label_handling and label_handling != "none_per_corpus" else None
    pairs = doc["pairs"] if logw is None else [p for p, w in zip(doc["pairs"], logw) if w is not None]
    labels = [LABEL2ID[p["label"]] for p in pairs]
    if label_handling == "none_per_corpus" and corpus is not None:
        labels = [len(LABELS) + NONE_SLOTS.index(corpus) if y == 0 else y for y in labels]
    novel = {frozenset((r["e1"], r["e2"])): r.get("novel") for r in doc.get("relations", [])}
    f = {
        "pmid": doc["pmid"],
        "input_ids": enc["input_ids"],
        "ent_pos": [[start2tok[mpos[mi]] for mi in ents[e]["mentions"]] for e in order],
        "pairs": [(index[p["e1"]], index[p["e2"]]) for p in pairs],
        "labels": labels,
        "meta": [(doc["pmid"], p["e1"], p["e2"], ents[p["e1"]]["type"], ents[p["e2"]]["type"], p["label"])
                 for p in pairs],
        "novelty": [NOVELTY.index(n) if (n := novel.get(frozenset((p["e1"], p["e2"])))) in NOVELTY else -100
                    for p in pairs],
        "unpairable": unpairable_gold(doc),
    }
    for k in AUX_CLASSES:
        f[k] = [p.get(k, -100) for p in pairs]
    if logw is not None:
        f["label_logw"] = [w for w in logw if w is not None]
    return f


def collate(feats: list[Feature], pad_id: int, device: torch.device | str = "cpu") -> dict[str, Any]:
    L = max(len(f["input_ids"]) for f in feats)
    ids = torch.full((len(feats), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(feats), L), dtype=torch.long)
    ent_pos, ent_mask, pairs, labels = [], [], [], []
    for b, f in enumerate(feats):
        n = len(f["input_ids"])
        ids[b, :n] = torch.tensor(f["input_ids"])
        mask[b, :n] = 1
        M = max((len(p) for p in f["ent_pos"]), default=1)
        pos = torch.zeros((len(f["ent_pos"]), M), dtype=torch.long)
        pm = torch.zeros((len(f["ent_pos"]), M), dtype=torch.bool)
        for i, p in enumerate(f["ent_pos"]):
            pos[i, :len(p)] = torch.tensor(p)
            pm[i, :len(p)] = True
        ent_pos.append(pos.to(device))
        ent_mask.append(pm.to(device))
        pairs.append(torch.tensor(f["pairs"], dtype=torch.long).view(-1, 2).to(device))
        labels.append(torch.tensor(f["labels"], dtype=torch.long).to(device))
    out = {"input_ids": ids.to(device), "attention_mask": mask.to(device),
           "ent_pos": ent_pos, "ent_mask": ent_mask, "pairs": pairs, "labels": labels}
    if "label_logw" in feats[0]:   # every feature of a label-handling run has it (KeyError otherwise)
        out["label_masks"] = [torch.tensor(f["label_logw"], dtype=torch.float).view(-1, len(LABELS)).to(device)
                              for f in feats]
    if "novelty" in feats[0]:
        out["novelty"] = [torch.tensor(f["novelty"], dtype=torch.long).to(device) for f in feats]
    for k in AUX_CLASSES:
        if k in feats[0]:
            out[k] = [torch.tensor(f[k], dtype=torch.long).to(device) for f in feats]
    if any("teacher" in f for f in feats):   # --distill: teacher probabilities, an all-zero row = no teacher
        rows = [p or [0.0] * len(LABELS) for f in feats for p in (f.get("teacher") or [None] * len(f["pairs"]))]
        out["teacher"] = torch.tensor(rows, dtype=torch.float).view(-1, len(LABELS)).to(device)
    return out


def atl_loss(logits: Tensor, y: Tensor) -> Tensor:
    """Adaptive-thresholding loss (Zhou et al. 2021) for a single label per pair, class 0 (None) = threshold.

    Positive y: CE over {None, y} plus CE of None over all classes except y. None pair: ordinary CE over all classes
    (its first term is a softmax over {None} alone, i.e. 0). Mean over pairs.
    """
    onehot = F.one_hot(y, logits.size(1)).bool()
    pos_vs_th = onehot.clone()
    pos_vs_th[:, 0] = True
    l1 = -logits.masked_fill(~pos_vs_th, float("-inf")).log_softmax(-1).gather(1, y[:, None]).squeeze(1)
    th_vs_neg = ~(onehot & (y > 0)[:, None])
    l2 = -logits.masked_fill(~th_vs_neg, float("-inf")).log_softmax(-1)[:, 0]
    return (l1 + l2).mean()


def marginal_loss(logits: Tensor, logw: Tensor) -> Tensor:
    """-log sum_k p_k exp(W_k), mean over pairs: W = 0/-inf is the set loss, W = log T the soft-T loss."""
    return (torch.logsumexp(logits, -1) - torch.logsumexp(logits + logw, -1)).mean()


def kd_loss(logits: Tensor, y: Tensor, teacher: Tensor, alpha: float = 0.5, temp: float = 2.0) -> Tensor:
    """Ensemble distillation, the preregistered form (§7 row 3): CE + alpha * KL(teacher_T || student_T), mean over pairs.

    Both distributions are softened with temperature ``temp``; no temp^2 factor and no (1 - alpha) on CE. Pairs without
    teacher probabilities (an all-zero row of ``teacher``) add no KL term. ``alpha`` 0 is exactly ``F.cross_entropy``.
    """
    ce = F.cross_entropy(logits, y)
    has = teacher.sum(-1) > 0
    if not bool(has.any()):
        return ce
    t = torch.softmax(teacher[has].clamp_min(1e-12).log() / temp, -1)
    kl = (t * (t.clamp_min(1e-12).log() - F.log_softmax(logits[has] / temp, -1))).sum(-1)
    return ce + alpha * kl.sum() / len(y)


def masked_group_norm(norm: nn.GroupNorm, x: Tensor, mask: Tensor) -> Tensor:
    """``norm`` on an [n, n, C] map with mean and variance over the cells where ``mask`` [n, n, 1] is 1 only (float32,
    like GroupNorm under autocast); masked cells come out zero. With an all-ones mask it equals ``norm``."""
    n, C, G = x.size(0), x.size(2), norm.num_groups
    xg, m = x.float().view(n, n, G, C // G), mask.float()[..., None]
    k = m.sum() * (C // G)
    mean = (xg * m).sum((0, 1, 3)) / k
    var = ((xg - mean[:, None]) ** 2 * m).sum((0, 1, 3)) / k
    y = ((xg - mean[:, None]) / (var[:, None] + norm.eps).sqrt()).view(n, n, C)
    return (y * norm.weight + norm.bias) * mask


class PairMatrix(nn.Module):
    """Entity-by-entity pair map (preregistration §7 row 1, improvement_plan §4.3 ``pmx``).

    ``F[s,o] = W[h_s; h_o; c_so]`` on candidate cells, symmetrised ``(F + F^T)/2``, non-candidate cells zeroed, then
    ``layers`` x residual blocks over the two entity axes: ``M = M + ReLU(GroupNorm(8)(Conv3x3(M)))``, then
    ``M = M + axial(M)`` (attention along rows, then along columns), the mask re-applied after each. The map is
    symmetrised once more before the candidate cells are gathered (conv kernels and attention are not symmetric,
    relations are undirected).

    Non-candidate cells never reach a candidate output: they are zero at every conv input, GroupNorm statistics are
    over candidate cells only (``masked_group_norm``) and attention keys are the candidate cells plus the diagonal
    (kept so no row is fully masked). Only the grid layout matters: the conv sees the entity order.

    One document at a time: nothing bigger than ``n_ent^2 x dim`` exists (42 entities x 128 = 0.9 MB).
    """

    def __init__(self, in_dim: int, dim: int, layers: int, heads: int = 8) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"pair_matrix_dim {dim} must be divisible by {heads} (GroupNorm groups, attention heads)")
        self.dim = dim
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(nn.ModuleDict({
            "conv": nn.Conv2d(dim, dim, 3, padding=1),
            "norm": nn.GroupNorm(heads, dim),
            "row": nn.MultiheadAttention(dim, heads, batch_first=True),
            "col": nn.MultiheadAttention(dim, heads, batch_first=True)}) for _ in range(layers))

    def forward(self, feat_so: Tensor, feat_os: Tensor, s: Tensor, o: Tensor, n_ent: int) -> Tensor:
        """``feat_*``: [n_pairs, in_dim] pair features of both directions; returns [n_pairs, dim] for the cells (s,o)."""
        M = feat_so.new_zeros(n_ent, n_ent, self.dim)
        M[s, o] = self.proj(feat_so).to(M.dtype)
        M[o, s] = self.proj(feat_os).to(M.dtype)
        cand = torch.zeros(n_ent, n_ent, dtype=torch.bool, device=M.device)
        cand[s, o] = cand[o, s] = True
        mask = cand[..., None].to(M.dtype)
        ignore = ~(cand | torch.eye(n_ent, dtype=torch.bool, device=M.device))   # key_padding_mask: True = not a key
        M = (M + M.transpose(0, 1)) / 2 * mask
        for b in self.blocks:
            conv = b["conv"](M.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
            M = (M + torch.relu(masked_group_norm(b["norm"], conv, mask))) * mask
            A = b["row"](M, M, M, key_padding_mask=ignore, need_weights=False)[0] * mask   # n rows of n cells
            At = A.transpose(0, 1)                                                          # n columns of n cells
            A = b["col"](At, At, At, key_padding_mask=ignore.transpose(0, 1), need_weights=False)[0].transpose(0, 1)
            M = (M + A) * mask
        return ((M + M.transpose(0, 1)) / 2 * mask)[s, o]


class DocRE(nn.Module):
    HPARAMS = ("context_pool", "block", "max_len", "stride", "none_weight", "pad_id", "loss", "n_extra_none",
               "pair_matrix_layers", "pair_matrix_dim", "nv_weight", "aux_heads", "aux_weight")

    def __init__(self, encoder: PreTrainedModel, num_labels: int = len(LABELS), context_pool: bool = True,
                 block: int = 64, max_len: int = 512, stride: int = 256, none_weight: float = 1.0,
                 pad_id: int = 0, loss: str = "ce", n_extra_none: int = 0, pair_matrix_layers: int = 0,
                 pair_matrix_dim: int = 128, nv_weight: float = 0.0, aux_heads: list[str] | tuple[str, ...] = (),
                 aux_weight: float = 1.0) -> None:
        """``n_extra_none`` = K per-corpus None classes (BioREx c1): K more outputs, predict uses the first 9.

        ``pair_matrix_layers`` > 0 adds the PairMatrix module (its gathered cell is concatenated to both head inputs);
        ``nv_weight`` > 0 adds the novelty head over the same bilinear pair representation. Both off by default.
        ``aux_heads`` (AUX_CLASSES keys) add one such head each, loss weight ``aux_weight``; none by default.
        """
        super().__init__()
        if loss not in ("ce", "atl") or (loss == "atl" and none_weight != 1.0):
            raise ValueError(f"loss {loss!r} with none_weight {none_weight}: expected ce, or atl with none_weight 1")
        self.loss, self.n_extra_none = loss, n_extra_none
        num_labels += n_extra_none
        hidden = encoder.config.hidden_size
        if hidden % block:
            raise ValueError(f"hidden size {hidden} must be divisible by block {block}")
        limit = getattr(encoder.config, "max_position_embeddings", max_len)
        if max_len > limit:
            raise ValueError(f"max_len {max_len} exceeds the backbone's {limit} position embeddings")
        self.encoder = encoder
        self.hidden, self.block = hidden, block
        self.context_pool, self.max_len, self.stride, self.none_weight, self.pad_id = \
            context_pool, max_len, stride, none_weight, pad_id
        in_dim = 2 * hidden if context_pool else hidden
        self.pair_matrix_layers, self.pair_matrix_dim, self.nv_weight = pair_matrix_layers, pair_matrix_dim, nv_weight
        # pair-matrix input = [h_s; h_o; c] (= in_dim + hidden, with or without the context vector)
        self.pmx = PairMatrix(in_dim + hidden, pair_matrix_dim, pair_matrix_layers) if pair_matrix_layers else None
        if self.pmx is not None:
            in_dim += pair_matrix_dim
        self.head_ext = nn.Linear(in_dim, hidden)
        self.tail_ext = nn.Linear(in_dim, hidden)
        self.bilinear = nn.Linear(hidden * block, num_labels)
        self.nv_head = nn.Linear(hidden * block, len(NOVELTY)) if nv_weight else None
        self.aux_heads, self.aux_weight = list(aux_heads), aux_weight
        self.aux = nn.ModuleDict({k: nn.Linear(hidden * block, AUX_CLASSES[k]) for k in self.aux_heads})
        w = torch.ones(num_labels)
        w[0] = none_weight
        self.register_buffer("class_weight", w)

    # ---- encoding -------------------------------------------------------------------------
    def _run(self, ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        out = self.encoder(input_ids=ids, attention_mask=mask, output_attentions=True)
        if not out.attentions:
            raise RuntimeError("encoder returned no attentions: load it with attn_implementation='eager'")
        return out.last_hidden_state.float(), out.attentions[-1].float().mean(1)   # [W,L,h], [W,L,L]

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Hidden states [B,L,h] and head-averaged last-layer attention [B,L,L]; exact windowing."""
        B, L = input_ids.shape
        if L <= self.max_len:
            return self._run(input_ids, attention_mask)
        lens = attention_mask.sum(1).tolist()
        w = self.max_len - 2
        wins: list[tuple[int, Tensor, Tensor]] = []   # (doc, global index per window token (-1=drop), ids)
        for b in range(B):
            n = int(lens[b])
            if n <= self.max_len:
                wins.append((b, torch.arange(n), input_ids[b, :n]))
                continue
            content, m = input_ids[b, 1:n - 1], n - 2
            starts = list(range(0, m - w, self.stride)) + [m - w]
            for k, s in enumerate(starts):
                ids = torch.cat([input_ids[b, :1], content[s:s + w], input_ids[b, n - 1:n]])
                g = torch.cat([torch.tensor([0 if k == 0 else -1]), torch.arange(s + 1, s + w + 1),
                               torch.tensor([n - 1 if k == len(starts) - 1 else -1])])
                wins.append((b, g, ids))
        wl = max(len(x[2]) for x in wins)
        ids = torch.full((len(wins), wl), self.pad_id, dtype=torch.long, device=input_ids.device)
        mask = torch.zeros((len(wins), wl), dtype=torch.long, device=input_ids.device)
        for i, (_, _, x) in enumerate(wins):
            ids[i, :len(x)] = x
            mask[i, :len(x)] = 1
        h, a = self._run(ids, mask)
        H = h.new_zeros((B, L, self.hidden))
        A = a.new_zeros((B, L, L))
        cnt = h.new_zeros((B, L))
        cntA = a.new_zeros((B, L, L))
        for i, (b, g, _) in enumerate(wins):
            g = g.to(h.device)
            keep = g >= 0
            gi, wi = g[keep], keep.nonzero().squeeze(1)
            H[b].index_add_(0, gi, h[i, wi])
            cnt[b].index_add_(0, gi, torch.ones_like(gi, dtype=h.dtype))
            A[b][gi[:, None], gi[None, :]] += a[i][wi[:, None], wi[None, :]]
            cntA[b][gi[:, None], gi[None, :]] += 1
        return H / cnt.clamp(min=1)[..., None], A / cntA.clamp(min=1)

    # ---- classification --------------------------------------------------------------------
    def _pair_feat(self, s_in: Tensor, o_in: Tensor) -> Tensor:
        zs, zo = torch.tanh(self.head_ext(s_in)), torch.tanh(self.tail_ext(o_in))
        b1 = zs.view(-1, self.hidden // self.block, self.block)
        b2 = zo.view(-1, self.hidden // self.block, self.block)
        return (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.hidden * self.block)

    def _score(self, s_in: Tensor, o_in: Tensor) -> Tensor:
        return self.bilinear(self._pair_feat(s_in, o_in))

    def forward(self, input_ids: Tensor, attention_mask: Tensor, ent_pos: list[Tensor], ent_mask: list[Tensor],
                pairs: list[Tensor], labels: list[Tensor] | None = None, label_masks: list[Tensor] | None = None,
                novelty: list[Tensor] | None = None, pc: list[Tensor] | None = None,
                ev: list[Tensor] | None = None) -> dict[str, Tensor]:
        """``label_masks`` (per doc [n_pairs, 9] log-weights W): loss = mean(logsumexp(z) - logsumexp(z + W)), the
        marginal likelihood of the admissible labels (one-hot W = cross-entropy); plain CE only, 9 outputs.
        ``novelty`` (per doc [n_pairs], NOVELTY index or -100): novelty-head cross-entropy, weighted ``nv_weight``,
        added to the loss; ignored when the head is off. ``pc`` / ``ev`` (per doc [n_pairs], class or -100): the same
        for each aux head (``<k>_logits``, ``<k>_loss``), weighted ``aux_weight``."""
        if label_masks is not None and (self.loss != "ce" or self.none_weight != 1.0 or self.n_extra_none):
            raise ValueError("label_masks need loss ce, none_weight 1 and no extra None classes")
        H, A = self.encode(input_ids, attention_mask)
        hs_l, ho_l = [], []
        for b, pr in enumerate(pairs):
            if pr.numel() == 0:
                continue
            Hb, pos, msk = H[b], ent_pos[b], ent_mask[b]                     # pos/msk: [n_ent, M]
            ent_emb = torch.logsumexp(Hb[pos].masked_fill(~msk[..., None], float("-inf")), dim=1)
            s, o = pr[:, 0], pr[:, 1]
            hs, ho = ent_emb[s], ent_emb[o]
            c = None
            if self.context_pool:
                ent_att = (A[b][pos] * msk[..., None]).sum(1) / msk.sum(1, keepdim=True)   # [n_ent, L]
                q = ent_att[s] * ent_att[o]
                c = (q / (q.sum(-1, keepdim=True) + 1e-10)) @ Hb                          # [n_pairs, h]
            if self.pmx is not None:                                          # one document at a time: n_ent^2 x dim
                cc = [c] if c is not None else []
                g = self.pmx(torch.cat([hs, ho, *cc], -1), torch.cat([ho, hs, *cc], -1), s, o, ent_emb.size(0))
                hs, ho = torch.cat([hs, *cc, g], -1), torch.cat([ho, *cc, g], -1)
            elif c is not None:
                hs, ho = torch.cat([hs, c], -1), torch.cat([ho, c], -1)
            hs_l.append(hs)
            ho_l.append(ho)
        nv_logits = None
        if not hs_l:
            logits = self.bilinear.weight.new_zeros((0, self.bilinear.out_features))
            if self.nv_head is not None:
                nv_logits = self.nv_head.weight.new_zeros((0, len(NOVELTY)))
            aux_logits = {k: h.weight.new_zeros((0, h.out_features)) for k, h in self.aux.items()}
        elif self.nv_head is None and not self.aux:
            hs, ho = torch.cat(hs_l), torch.cat(ho_l)
            logits = (self._score(hs, ho) + self._score(ho, hs)) / 2
            aux_logits = {}
        else:
            hs, ho = torch.cat(hs_l), torch.cat(ho_l)
            f_so, f_os = self._pair_feat(hs, ho), self._pair_feat(ho, hs)     # same pair representation, all heads
            logits = (self.bilinear(f_so) + self.bilinear(f_os)) / 2
            if self.nv_head is not None:
                nv_logits = (self.nv_head(f_so) + self.nv_head(f_os)) / 2
            aux_logits = {k: (h(f_so) + h(f_os)) / 2 for k, h in self.aux.items()}
        out = {"logits": logits}
        if nv_logits is not None:
            out["nv_logits"] = nv_logits
        out.update({f"{k}_logits": z for k, z in aux_logits.items()})
        if labels is not None:
            y = torch.cat(labels)
            if not y.numel():
                out["loss"] = self.bilinear.weight.sum() * 0.0   # keeps a grad_fn for a zero-pair batch
            elif label_masks is not None:
                out["loss"] = marginal_loss(logits.float(), torch.cat(label_masks))
            elif self.loss == "atl":
                out["loss"] = atl_loss(logits.float(), y)
            else:
                out["loss"] = F.cross_entropy(logits.float(), y, weight=self.class_weight)
            if nv_logits is not None and novelty is not None:
                nvy = torch.cat(novelty)
                if bool((nvy >= 0).any()):                                    # gold relations only (-100 = ignored)
                    out["nv_loss"] = F.cross_entropy(nv_logits.float(), nvy, ignore_index=-100)
                    out["loss"] = out["loss"] + self.nv_weight * out["nv_loss"]
            for k, ys in (("pc", pc), ("ev", ev)):
                if k not in aux_logits or ys is None:
                    continue
                y_k = torch.cat(ys)
                if bool((y_k >= 0).any()):                                    # -100 = no aux label
                    out[f"{k}_loss"] = F.cross_entropy(aux_logits[k].float(), y_k, ignore_index=-100)
                    out["loss"] = out["loss"] + self.aux_weight * out[f"{k}_loss"]
        return out

    # ---- persistence -----------------------------------------------------------------------
    def hparams(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.HPARAMS}

    def save(self, out_dir: str | Path, tokenizer: PreTrainedTokenizerBase) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out_dir)
        self.encoder.config.save_pretrained(out_dir)
        json.dump(self.hparams(), open(out_dir / "re_config.json", "w"), indent=1)
        torch.save(self.state_dict(), out_dir / "model.pt")

    @classmethod
    def load(cls, run_dir: str | Path, device: torch.device | str = "cpu") -> tuple["DocRE", PreTrainedTokenizerBase]:
        run_dir = Path(run_dir)
        tok = AutoTokenizer.from_pretrained(run_dir)
        cfg = AutoConfig.from_pretrained(run_dir)
        enc = AutoModel.from_config(cfg, attn_implementation="eager")
        enc.resize_token_embeddings(len(tok), mean_resizing=False)
        model = cls(enc, **json.load(open(run_dir / "re_config.json")))
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        return model.to(device).eval(), tok


def build_model(backbone: str, **hparams: Any) -> tuple[DocRE, PreTrainedTokenizerBase]:
    """Pretrained backbone (HF id or dir) + markers + fresh head."""
    tok = AutoTokenizer.from_pretrained(backbone)
    enc = AutoModel.from_pretrained(backbone, attn_implementation="eager")
    add_markers(tok, enc)
    return DocRE(enc, pad_id=tok.pad_token_id or 0, **hparams), tok


@torch.no_grad()
def predict(model: DocRE, feats: list[Feature], batch_docs: int = 4, device: torch.device | str | None = None,
            autocast: Any = None) -> list[dict]:
    """Rows {pmid,e1,e2,t1,t2,gold,pred,probs{label:p}} for every candidate pair, in input order.

    With the novelty head on, every row also carries ``gold_novel`` (None when the pair has no gold relation),
    ``probs_novel`` {No, Novel: p} and ``pred_novel`` (its argmax) — evaluate.all_metrics then reports the +NV metric
    and ensemble.ensemble_rows averages ``probs_novel``. With aux heads, ``probs_<k>`` = [p per class] for each.
    ``device`` defaults to the model's own device.
    """
    model.eval()
    device = next(model.parameters()).device if device is None else device
    nv = model.nv_head is not None
    rows: list[dict] = []
    for i in range(0, len(feats), batch_docs):
        chunk = feats[i:i + batch_docs]
        batch = collate(chunk, model.pad_id, device)
        batch.pop("labels")
        batch.pop("label_masks", None)
        batch.pop("teacher", None)
        with (autocast or contextlib.nullcontext()):
            out = model(**batch)
        logits = out["logits"][:, :len(LABELS)]                         # c1: extra None classes are training-only
        probs = torch.softmax(logits.float(), -1).cpu().tolist()
        metas = [m for f in chunk for m in f["meta"]]
        novel = list(zip([g for f in chunk for g in f["novelty"]],
                         torch.softmax(out["nv_logits"].float(), -1).cpu().tolist())) if nv else [()] * len(metas)
        aux = {k: torch.softmax(out[f"{k}_logits"].float(), -1).cpu().tolist() for k in model.aux}
        for j, ((pmid, e1, e2, t1, t2, gold), p, n) in enumerate(zip(metas, probs, novel)):
            row = {"pmid": pmid, "e1": e1, "e2": e2, "t1": t1, "t2": t2, "gold": gold,
                   "pred": LABELS[max(range(len(p)), key=p.__getitem__)], "probs": dict(zip(LABELS, p))}
            if nv:
                row["gold_novel"] = NOVELTY[n[0]] if n[0] >= 0 else None
                row["probs_novel"] = dict(zip(NOVELTY, n[1]))
                row["pred_novel"] = max(NOVELTY, key=row["probs_novel"].__getitem__)
            row.update({f"probs_{k}": pk[j] for k, pk in aux.items()})
            rows.append(row)
    return rows
