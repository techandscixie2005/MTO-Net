import torch
import numpy as np

def extract_mto_contributions(model, z, pos, batch, top_r=5):
    model.eval()
    with torch.no_grad():
        out = model(z=z, pos=pos, batch=batch)
        coeff = out.get("coeff")
        O = out.get("O")
    if coeff is None:
        raise ValueError("Model must return coeff")
    B = coeff.shape[0]
    contribs = {}
    for b in range(B):
        c = coeff[b]
        non_zero = (c.abs().sum(dim=-1) > 0)
        c_valid = c[non_zero]
        k = min(top_r, c_valid.shape[0])
        if k > 0:
            contribs[b] = c_valid[:k].cpu().numpy()
        else:
            contribs[b] = np.zeros((0, c.shape[-1]))
    return {"contributions": contribs, "O": O.cpu().numpy() if O is not None else None}

def subspace_similarity(Q_a, Q_b, r):
    r = min(r, Q_a.shape[0], Q_b.shape[0])
    if r == 0:
        return 0.0
    a = Q_a[:r].reshape(r, -1)
    b = Q_b[:r].reshape(r, -1)
    if a.shape[1] > r:
        Qa, _ = np.linalg.qr(a.T)
    else:
        Qa = a
    if b.shape[1] > r:
        Qb, _ = np.linalg.qr(b.T)
    else:
        Qb = b
    try:
        Qa_ortho, _ = np.linalg.qr(Qa)
        Qb_ortho, _ = np.linalg.qr(Qb)
        proj = Qa_ortho.T @ Qb_ortho
        s = np.trace(proj @ proj.T) / r
        return float(np.clip(s, 0.0, 1.0))
    except np.linalg.LinAlgError:
        return 0.0

def seed_subspace_stability(contrib_maps, top_r=5):
    seeds = sorted(contrib_maps.keys())
    results = []
    for i, sa in enumerate(seeds):
        for j, sb in enumerate(seeds):
            if j <= i:
                continue
            mols_a = set(contrib_maps[sa]["contributions"].keys())
            mols_b = set(contrib_maps[sb]["contributions"].keys())
            common = mols_a & mols_b
            if not common:
                continue
            sims = []
            for mol in common:
                Qa = contrib_maps[sa]["contributions"][mol]
                Qb = contrib_maps[sb]["contributions"][mol]
                sims.append(subspace_similarity(Qa, Qb, top_r))
            results.append({
                "seed_a": sa, "seed_b": sb,
                "similarity": float(np.mean(sims)),
                "n_mols": len(common),
            })
    return {"pairs": results, "top_r": top_r}

def stage_stability(contribs_a, contribs_b):
    common = set(contribs_a["contributions"].keys()) & set(contribs_b["contributions"].keys())
    corrs = []
    for mol in common:
        wa = contribs_a["contributions"][mol].reshape(-1)
        wb = contribs_b["contributions"][mol].reshape(-1)
        min_len = min(len(wa), len(wb))
        if min_len > 1:
            corr = np.corrcoef(wa[:min_len], wb[:min_len])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
    return {"mean_correlation": float(np.mean(corrs)) if corrs else 0.0, "n_mols": len(corrs)}
