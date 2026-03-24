"""
Genetic Algorithm Loss Evolver — evolve loss functions via crossover + mutation.

Each loss is a genome: a weighted combination of primitive terms.
GA operates on structured genomes, not raw code strings.

Genome = {
    "terms": [
        {"name": "pnl", "weight": 1.0, "sign": -1},  # maximize
        {"name": "sortino", "weight": 0.5, "sign": -1},
        ...
    ],
    "curriculum": {"enabled": True, "rate": 0.2},
    "tanh_temp": 100.0,
}
"""
import copy
import random
import math
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F


# ── Primitive Loss Terms ─────────────────────────────────

def _velocity(v_pred, v_target, **kw):
    """Base flow matching loss. Always included, not evolvable."""
    return F.mse_loss(v_pred, v_target)


def _pnl(x_pred, x_real, tanh_temp=100.0, **kw):
    """Differentiable PnL: tanh(T*pred) * real."""
    return torch.mean(torch.tanh(tanh_temp * x_pred) * x_real)


def _mse(x_pred, x_real, **kw):
    """Prediction MSE."""
    return F.mse_loss(x_pred, x_real)


def _sharpe(x_pred, x_real, tanh_temp=100.0, **kw):
    """Mini-batch Sharpe ratio."""
    pnl = torch.tanh(tanh_temp * x_pred) * x_real
    return pnl.mean() / (pnl.std(unbiased=False) + 1e-8)


def _sortino(x_pred, x_real, tanh_temp=100.0, **kw):
    """Sortino ratio (downside risk only)."""
    pnl = torch.tanh(tanh_temp * x_pred) * x_real
    downside = torch.sqrt(F.relu(-pnl).pow(2).mean() + 1e-8)
    return pnl.mean() / (downside + 1e-4)


def _std(x_pred, x_real, tanh_temp=100.0, **kw):
    """PnL volatility."""
    pnl = torch.tanh(tanh_temp * x_pred) * x_real
    return pnl.std(unbiased=False)


def _huber_pnl(x_pred, x_real, tanh_temp=100.0, **kw):
    """Huber loss on PnL (robust to outliers)."""
    pnl = torch.tanh(tanh_temp * x_pred) * x_real
    return F.huber_loss(pnl, torch.zeros_like(pnl), delta=0.5)


def _directional_acc(x_pred, x_real, tanh_temp=100.0, **kw):
    """Soft directional accuracy: sigmoid(T * pred * real)."""
    return torch.mean(torch.sigmoid(tanh_temp * x_pred * x_real))


def _winrate_margin(x_pred, x_real, tanh_temp=100.0, **kw):
    """Hinge loss on correct direction: max(0, margin - pred*real)."""
    margin = 0.01
    return torch.mean(F.relu(margin - x_pred * x_real))


def _cond_momentum(x_pred, x_real, condition, **kw):
    """Momentum-aware: weight PnL by recent trend strength."""
    if condition.shape[1] < 3:
        return _pnl(x_pred, x_real, **kw)
    trend = condition[:, -3:].mean(dim=1, keepdim=True)
    trend_strength = torch.abs(trend)
    pnl = torch.tanh(100.0 * x_pred) * x_real
    return torch.mean(pnl * (1.0 + trend_strength))


def _cond_vol_scale(x_pred, x_real, condition, **kw):
    """Volatility-scaled PnL: scale by inverse of recent vol."""
    if condition.shape[1] < 5:
        return _pnl(x_pred, x_real, **kw)
    recent_vol = condition[:, -5:].std(dim=1, keepdim=True) + 1e-6
    pnl = torch.tanh(100.0 * x_pred) * x_real
    return torch.mean(pnl / recent_vol)


def _quantile_asym(x_pred, x_real, **kw):
    """Asymmetric quantile loss: penalize underprediction of positive returns more."""
    tau = 0.6  # bias toward positive returns
    error = x_real - x_pred
    return torch.mean(torch.where(error >= 0, tau * error.abs(), (1 - tau) * error.abs()))


TERM_REGISTRY = {
    "pnl": _pnl,
    "mse": _mse,
    "sharpe": _sharpe,
    "sortino": _sortino,
    "std": _std,
    "huber_pnl": _huber_pnl,
    "directional_acc": _directional_acc,
    "winrate_margin": _winrate_margin,
    "cond_momentum": _cond_momentum,
    "cond_vol_scale": _cond_vol_scale,
    "quantile_asym": _quantile_asym,
}

# Terms that should be maximized (sign = -1 in loss)
MAXIMIZE_TERMS = {"pnl", "sharpe", "sortino", "directional_acc", "cond_momentum", "cond_vol_scale"}
# Terms that should be minimized (sign = +1 in loss)
MINIMIZE_TERMS = {"mse", "std", "huber_pnl", "winrate_margin", "quantile_asym"}


# ── Genome ───────────────────────────────────────────────

def random_genome(n_terms: int = 3) -> dict:
    """Generate a random genome."""
    all_terms = list(TERM_REGISTRY.keys())
    selected = random.sample(all_terms, min(n_terms, len(all_terms)))

    terms = []
    for name in selected:
        sign = -1 if name in MAXIMIZE_TERMS else 1
        weight = round(random.uniform(0.1, 2.0), 2)
        terms.append({"name": name, "weight": weight, "sign": sign})

    return {
        "terms": terms,
        "curriculum": {
            "enabled": random.random() < 0.5,
            "rate": round(random.uniform(0.05, 0.5), 2),
        },
        "tanh_temp": random.choice([50.0, 100.0, 200.0]),
    }


def genome_to_loss_fn(genome: dict):
    """Convert genome to a callable loss function."""
    terms = genome["terms"]
    curriculum = genome["curriculum"]
    tanh_temp = genome.get("tanh_temp", 100.0)

    def loss_fn(v_pred, v_target, x_pred, x_real, condition, epoch):
        vel = F.mse_loss(v_pred, v_target)

        if curriculum["enabled"]:
            fin_weight = 1.0 - math.exp(-curriculum["rate"] * max(0, epoch))
        else:
            fin_weight = 1.0

        fin_loss = torch.tensor(0.0, device=v_pred.device)
        for t in terms:
            fn = TERM_REGISTRY[t["name"]]
            val = fn(x_pred=x_pred, x_real=x_real, condition=condition,
                     v_pred=v_pred, v_target=v_target, tanh_temp=tanh_temp)
            fin_loss = fin_loss + t["sign"] * t["weight"] * val

        return vel + fin_weight * fin_loss

    return loss_fn


def genome_to_name(genome: dict) -> str:
    """Generate a descriptive name from genome."""
    parts = []
    if genome["curriculum"]["enabled"]:
        parts.append("curr")
    for t in genome["terms"]:
        parts.append(t["name"].replace("cond_", "c"))
    return "_".join(parts)[:40]


def genome_to_description(genome: dict) -> str:
    """Human-readable description."""
    terms_str = ", ".join(
        f"{'max' if t['sign'] < 0 else 'min'} {t['name']}(w={t['weight']})"
        for t in genome["terms"]
    )
    curr = f"curriculum(rate={genome['curriculum']['rate']})" if genome["curriculum"]["enabled"] else "no-curriculum"
    return f"{terms_str} | {curr} | tanh={genome.get('tanh_temp', 100)}"


# ── GA Operators ─────────────────────────────────────────

def crossover(parent_a: dict, parent_b: dict) -> dict:
    """Single-point crossover on term lists + parameter blend."""
    child = copy.deepcopy(parent_a)

    # Crossover terms: take some from each parent
    all_terms = parent_a["terms"] + parent_b["terms"]
    # Deduplicate by name, prefer higher-fitness parent's weight
    seen = {}
    for t in all_terms:
        if t["name"] not in seen:
            seen[t["name"]] = copy.deepcopy(t)
        else:
            # Blend weights
            seen[t["name"]]["weight"] = round(
                0.5 * seen[t["name"]]["weight"] + 0.5 * t["weight"], 2
            )

    # Random subset of 1-4 terms
    n = random.randint(1, max(1, min(4, len(seen))))
    child["terms"] = random.sample(list(seen.values()), n)

    # Blend curriculum
    if random.random() < 0.5:
        child["curriculum"] = copy.deepcopy(parent_b["curriculum"])

    # Blend tanh_temp
    child["tanh_temp"] = random.choice([parent_a.get("tanh_temp", 100), parent_b.get("tanh_temp", 100)])

    return child


def mutate(genome: dict, mutation_rate: float = 0.3) -> dict:
    """Mutate genome: modify weights, add/remove terms, flip curriculum."""
    g = copy.deepcopy(genome)

    # Mutate weights
    for t in g["terms"]:
        if random.random() < mutation_rate:
            t["weight"] = round(t["weight"] * random.uniform(0.5, 2.0), 2)
            t["weight"] = max(0.01, min(5.0, t["weight"]))

    # Add a random term
    if random.random() < mutation_rate * 0.5:
        existing = {t["name"] for t in g["terms"]}
        available = [n for n in TERM_REGISTRY if n not in existing]
        if available:
            name = random.choice(available)
            sign = -1 if name in MAXIMIZE_TERMS else 1
            g["terms"].append({
                "name": name,
                "weight": round(random.uniform(0.1, 1.5), 2),
                "sign": sign,
            })

    # Remove a random term (keep at least 1)
    if len(g["terms"]) > 1 and random.random() < mutation_rate * 0.3:
        g["terms"].pop(random.randint(0, len(g["terms"]) - 1))

    # Flip curriculum
    if random.random() < mutation_rate * 0.3:
        g["curriculum"]["enabled"] = not g["curriculum"]["enabled"]

    # Mutate curriculum rate
    if g["curriculum"]["enabled"] and random.random() < mutation_rate:
        g["curriculum"]["rate"] = round(
            g["curriculum"]["rate"] * random.uniform(0.5, 2.0), 2
        )
        g["curriculum"]["rate"] = max(0.01, min(1.0, g["curriculum"]["rate"]))

    # Mutate tanh_temp
    if random.random() < mutation_rate * 0.2:
        g["tanh_temp"] = random.choice([25.0, 50.0, 100.0, 200.0, 500.0])

    return g


def tournament_select(population: list[dict], fitnesses: list[float], k: int = 3) -> dict:
    """Tournament selection: pick k random, return the best."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])


# ── Seed Population from FinGAN Baselines ────────────────

def fingan_seed_genomes() -> list[dict]:
    """Create initial population from FinGAN's 8 known loss combos."""
    seeds = [
        # velocity_only (implicit baseline)
        {"terms": [], "curriculum": {"enabled": False, "rate": 0}, "tanh_temp": 100},
        # fingan_pnl
        {"terms": [{"name": "pnl", "weight": 1.0, "sign": -1}],
         "curriculum": {"enabled": False, "rate": 0}, "tanh_temp": 100},
        # fingan_pnl_sr (best baseline)
        {"terms": [{"name": "pnl", "weight": 1.0, "sign": -1},
                   {"name": "sharpe", "weight": 1.0, "sign": -1}],
         "curriculum": {"enabled": False, "rate": 0}, "tanh_temp": 100},
        # curriculum_sortino_pnl (current champion)
        {"terms": [{"name": "pnl", "weight": 1.0, "sign": -1},
                   {"name": "sortino", "weight": 0.5, "sign": -1}],
         "curriculum": {"enabled": True, "rate": 0.2}, "tanh_temp": 100},
        # fingan_pnl_std
        {"terms": [{"name": "pnl", "weight": 1.0, "sign": -1},
                   {"name": "std", "weight": 1.0, "sign": 1}],
         "curriculum": {"enabled": False, "rate": 0}, "tanh_temp": 100},
        # fingan_sr
        {"terms": [{"name": "sharpe", "weight": 1.0, "sign": -1}],
         "curriculum": {"enabled": False, "rate": 0}, "tanh_temp": 100},
    ]
    return seeds


# ── Main GA Loop ─────────────────────────────────────────

def run_ga(
    evaluate_fn,   # (loss_fn, name) -> float (mean SR)
    pop_size: int = 12,
    n_generations: int = 10,
    elite_k: int = 2,
    mutation_rate: float = 0.3,
    seed: int = 42,
    results_dir: str = "results",
    verbose: bool = True,
) -> list[dict]:
    """
    Run genetic algorithm to evolve loss functions.

    evaluate_fn(loss_fn, name) -> float: Stage 1 mean SR
    Returns: sorted list of {genome, name, fitness, generation}
    """
    random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize population: seeds + random
    population = fingan_seed_genomes()
    while len(population) < pop_size:
        population.append(random_genome(n_terms=random.randint(2, 4)))

    all_results = []

    for gen in range(1, n_generations + 1):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  GENERATION {gen}/{n_generations} (pop={len(population)})")
            print(f"{'#'*60}")

        # Evaluate population
        fitnesses = []
        for i, genome in enumerate(population):
            name = f"gen{gen}_{genome_to_name(genome)}_{i}"
            desc = genome_to_description(genome)
            loss_fn = genome_to_loss_fn(genome)

            if verbose:
                print(f"\n[{i+1}/{len(population)}] {name}")
                print(f"  {desc}")

            try:
                sr = evaluate_fn(loss_fn, name)
            except Exception as e:
                print(f"  FAILED: {e}")
                sr = -999.0

            fitnesses.append(sr)

            result = {
                "generation": gen,
                "index": i,
                "name": name,
                "fitness": sr,
                "genome": genome,
                "description": desc,
            }
            all_results.append(result)

            if verbose:
                print(f"  SR = {sr:.4f}")

        # Log generation summary
        ranked = sorted(zip(fitnesses, population, range(len(population))),
                        key=lambda x: -x[0])

        if verbose:
            print(f"\n--- Generation {gen} Leaderboard ---")
            for rank, (fit, genome, idx) in enumerate(ranked[:5]):
                print(f"  {rank+1}. SR={fit:.4f} | {genome_to_name(genome)}")

        # Save generation results
        gen_path = os.path.join(results_dir, f"ga_gen_{gen:03d}.json")
        with open(gen_path, "w") as f:
            json.dump({
                "generation": gen,
                "results": [
                    {"name": all_results[-(len(population)-i)]["name"],
                     "fitness": fitnesses[i],
                     "genome": population[i],
                     "description": genome_to_description(population[i])}
                    for i in range(len(population))
                ],
                "best_fitness": ranked[0][0],
                "best_genome": ranked[0][1],
            }, f, indent=2, default=str)

        # Selection + reproduction
        # Elite: top-k survive unchanged
        new_pop = [copy.deepcopy(ranked[i][1]) for i in range(min(elite_k, len(ranked)))]

        # Fill rest via tournament selection + crossover + mutation
        while len(new_pop) < pop_size:
            if random.random() < 0.7:
                # Crossover
                p1 = tournament_select(population, fitnesses)
                p2 = tournament_select(population, fitnesses)
                child = crossover(p1, p2)
            else:
                # Clone + mutate
                child = tournament_select(population, fitnesses)

            child = mutate(child, mutation_rate)
            new_pop.append(child)

        population = new_pop

    # Final ranking
    all_results.sort(key=lambda x: -x["fitness"])
    return all_results
