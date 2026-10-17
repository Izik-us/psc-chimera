"""Small, dependency-free contracts used by merge-readiness checks."""


def effective_retrieval_k(requested: int, index_size: int) -> int:
    if requested < 1:
        raise ValueError("requested k must be positive")
    if index_size < 0:
        raise ValueError("index size cannot be negative")
    return min(requested, index_size)


def unique_conflict_pairs(pairs):
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs if i != j}


def scalarize_objectives(values, weights):
    if len(values) != len(weights):
        raise ValueError("objective/weight dimensions must match")
    if not values:
        raise ValueError("at least one objective is required")
    total = sum(float(w) for w in weights)
    if total <= 0:
        raise ValueError("objective weights must have positive total")
    return sum(float(v) * float(w) for v, w in zip(values, weights)) / total
