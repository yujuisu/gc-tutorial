import numpy as np
from functools import reduce
from itertools import combinations
from kingdon import Algebra, MultiVector


def create_random_multivector(alg: Algebra, normalized=False):
    n = len(alg.blades.blades.keys())
    vec = np.random.random(n)
    if normalized:
        vec = vec / (vec**2).sum()
    return alg.multivector(vec)


def max_diff(A, B):
    return np.max(np.abs((A - B)[:]))


def assert_diff(A, B, err=1e-10):
    assert max_diff(A, B) < err


def even_grades(A):
    return A.grade(A.grades[::2])


def odd_grades(A):
    return A.grade(A.grades[1::2])


def inner(A, B):
    if isinstance(A, MultiVector) and isinstance(B, MultiVector):
        return (A - A.grade(0)) | (B - B.grade(0))
    return 0


def create_r_vectors(r, alg):
    return [create_random_multivector(alg).grade(1) for _ in range(r)]


def cyclic_reorder(_v, k):
    return _v[k:] + _v[:k]


def cross(A, B):
    return (A * B - B * A) / 2


def wedge(vectors) -> MultiVector:
    if len(vectors) == 0:
        return 1
    return reduce(lambda a, b: a ^ b, vectors)


def gprod(vectors) -> MultiVector:
    if len(vectors) == 0:
        return 1
    return reduce(lambda a, b: a * b, vectors)


def create_r_blade(r, alg):
    return wedge(create_r_vectors(r, alg))


def P(a, A):
    return ((a | A) | A) / A**2


def max_grade(B):
    for r in B.grades[::-1]:
        Br = B.grade(r)
        if np.max(np.abs(Br[:])) > 1e-8:
            return Br


def gram_schmidt(vectors):
    o_vecs = [vectors[0]]
    Ar = vectors[0]
    for v in vectors[1:]:
        Ar1 = Ar ^ v
        o_vecs.append(Ar | Ar1)
        Ar = Ar1
    return o_vecs


def reciprocal(vectors):
    An = wedge(vectors)
    n = len(vectors)
    return [
        (-1) ** (k + n - 1) * (An | wedge(vectors[:k] + vectors[k + 1:])) / An**2
        for k in range(n)
    ]


def multiindex(n, r):
    return list(combinations(range(n), r))


def r_indexes(n, r):
    return list(combinations(range(n), r))


def extract(vectors, indexes):
    return [vectors[i] for i in indexes]


def r_vector_frame(vectors, r, reverse=False):
    n = len(vectors)
    indexes = r_indexes(n, r)
    if reverse:
        return [wedge(extract(vectors, i[::-1])) for i in indexes]
    return [wedge(extract(vectors, i)) for i in indexes]


def find_complement(combo, full):
    return [item for item in full if item not in combo]


def comp_indexes(indexes, full):
    return [find_complement(comb, full) for comb in indexes]


def multi_frame(vectors, reverse=False):
    # reverse for reciprocal frames
    multi_frame = [1]
    for r in range(1, len(vectors) + 1):
        multi_frame += r_vector_frame(vectors, r, reverse)
    return multi_frame


def reci_frame(vectors):
    return multi_frame(reciprocal(vectors), reverse=True)


def normsq(X):
    return X.sp(X.reverse())[0]
