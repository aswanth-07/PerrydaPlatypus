"""Same-sign sets: glosses that WLASL's own index shows to be one sign.

Two glosses are linked when one source video segment (same URL, overlapping frame range) is listed under
both. Links are closed into sets. Only the listings of the given splits are used, so a test listing never
creates a link: sets for a model trained on train+val use train and val listings, never test.

The answer rule keeps the model's probability for each set and splits it between the set's glosses by a
blend of the model's own split and how often WLASL lists each gloss:

    p'(g) = P(set of g) * p(g)^gamma * n(g)^beta / sum over the set of p(h)^gamma * n(h)^beta

gamma and beta were chosen on WLASL val with a model that never trained on val (scripts/set_rule.py).
"""
import collections

import torch

GAMMA, BETA = 0.25, 2.0


def same_sign_sets(entries, n, splits=("train", "val"), min_overlap=0.5):
    """entries: the WLASL index. Returns (set id per gloss as a LongTensor of n, number of links)."""
    by_url = collections.defaultdict(list)
    for c, e in enumerate(entries[:n]):
        for i in e["instances"]:
            if i["split"] in splits:
                by_url[i["url"]].append((c, i["frame_start"], i["frame_end"]))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    links = set()
    for items in by_url.values():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (ca, sa, ea), (cb, sb, eb) = items[a], items[b]
                if ca == cb:
                    continue
                if ea != -1 and eb != -1:  # both are cuts of a longer video: they must overlap
                    inter, union = min(ea, eb) - max(sa, sb), max(ea, eb) - min(sa, sb)
                    if inter <= 0 or inter / max(union, 1) < min_overlap:
                        continue
                links.add((min(ca, cb), max(ca, cb)))
    for a, b in links:
        parent[find(a)] = find(b)
    _, set_of = torch.unique(torch.tensor([find(c) for c in range(n)]), return_inverse=True)
    return set_of, len(links)


def wlasl_counts(entries, n, splits=("train", "val")):
    """Clips WLASL lists per gloss in the given splits (the gloss's weight inside its set)."""
    return torch.tensor([sum(1 for i in e["instances"] if i["split"] in splits) for e in entries[:n]], dtype=torch.float32)


def set_mass(p, set_of):
    """p: B x n class probabilities -> B x n, each gloss carrying the summed probability of its whole set."""
    mass = torch.zeros(p.shape[0], int(set_of.max()) + 1, dtype=p.dtype, device=p.device).index_add_(1, set_of, p)
    return mass[:, set_of]


def answer_rule(p, set_of, counts, gamma=GAMMA, beta=BETA):
    w = p.clamp_min(1e-12).pow(gamma) * counts.to(p).clamp_min(0.5).pow(beta)[None]
    return set_mass(p, set_of) * w / set_mass(w, set_of)


class SetRule(torch.nn.Module):
    """The answer rule as graph ops (Gather and ReduceSum over a padded member table), for the exported model.
    forward(p) -> (p', set probability per gloss)."""

    def __init__(self, set_of, counts, gamma=GAMMA, beta=BETA):
        super().__init__()
        n, k = len(set_of), int(torch.bincount(set_of).max())
        members = torch.full((int(set_of.max()) + 1, k), n, dtype=torch.long)   # index n is a zero pad column
        fill = [0] * members.shape[0]
        for g, s in enumerate(set_of.tolist()):
            members[s, fill[s]] = g
            fill[s] += 1
        self.gamma = gamma
        self.register_buffer("members", members.reshape(-1), persistent=False)
        self.register_buffer("set_of", set_of.clone(), persistent=False)
        self.register_buffer("weight", counts.clamp_min(0.5).pow(beta), persistent=False)
        self.shape = members.shape

    def _mass(self, v):
        pad = torch.cat([v, torch.zeros_like(v[:, :1])], 1)
        per_set = pad.index_select(1, self.members).reshape(v.shape[0], *self.shape).sum(2)
        return per_set.index_select(1, self.set_of)

    def forward(self, p):
        w = p.clamp_min(1e-12).pow(self.gamma) * self.weight
        mass = self._mass(p)
        return mass * w / self._mass(w), mass
