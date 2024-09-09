import numpy as np
from functools import reduce
from itertools import combinations, product, permutations
from kingdon import Algebra, MultiVector
from math import factorial


def create_random_multivector(alg: Algebra):
    n = len(alg.blades.blades.keys())
    vec = np.random.random(n)
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


def assert_simple(A, tol=1e-8):
    Asquare = A**2
    if Asquare[1:]:
        assert np.max(np.abs(Asquare[1:])) < tol, "not simple"
    return Asquare


def assert_not_simple(A, tol=1e-6):
    Asquare = A**2
    assert len(Asquare.grades) > 1, "simple"
    if Asquare[1:]:
        assert np.max(np.abs(Asquare[1:])) > tol, "simple"
    return Asquare


def P(a, A):  # projection of a onto a simple blade A
    Asquare = assert_simple(A)
    return (1/Asquare[0])*((a|A)|A)


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


# def reciprocal(vectors):
#     An = wedge(vectors)
#     n = len(vectors)
#     return [
#         (-1) ** (k + n - 1) * (An | wedge(vectors[:k] + vectors[k + 1:])) / An**2
#         for k in range(n)
#     ]


def reciprocal(blades):
    I = wedge(blades)
    dualblades = []
    for k in range(len(blades)):
        sign = (-1) ** (blades[k].grades[0] * sum(blade.grades[0] for blade in blades[:k]))
        dualblades.append(sign * wedge(blades[:k] + blades[k+1:]) * I.inv())
    return dualblades


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


def r_vector_frame_vectors(vectors, r, reverse=False):
    n = len(vectors)
    indexes = r_indexes(n, r)
    return [extract(vectors, i) for i in indexes]


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


def multi_frame_vectors(vectors, reverse=False):
    # reverse for reciprocal frames
    multi_frame = []
    for r in range(1, len(vectors) + 1):
        multi_frame += r_vector_frame_vectors(vectors, r, reverse)
    return multi_frame


def reci_frame(vectors):
    return multi_frame(reciprocal(vectors), reverse=True)


def normsq(X):
    return X.sp(X.reverse())[0]


def norm(A):
    return np.sqrt(normsq(A))


def differential(F, X, A, h=1e-6):
    d = (h)*A
    return (1/(2*h))*F(X+d) - (1/(2*h))*F(X-d)


def derivative(F, X, alg, h=1e-6, grade=None, frame=None, r_frame=None):
    if not frame:
        if grade or (grade == 0):
            frame = r_vector_frame(alg.frame, grade)
            r_frame = r_vector_frame(reciprocal(alg.frame), grade, reverse=True)
        else:
            frame = multi_frame(alg.frame)
            r_frame = reci_frame(alg.frame)
    dF = 0
    for v, r in zip(frame, r_frame):
        dF += r * (differential(F, X, v, h))
    return dF


def curl(F, X, alg, h=1e-6, grade=None, frame=None, r_frame=None):
    if not frame:
        if grade or (grade == 0):
            frame = r_vector_frame(alg.frame, grade)
            r_frame = r_vector_frame(reciprocal(alg.frame), grade, reverse=True)
        else:
            frame = multi_frame(alg.frame)
            r_frame = reci_frame(alg.frame)
    dF = 0
    for v, r in zip(frame, r_frame):
        dF += r ^ (differential(F, X, v, h))
    return dF


def adjoint(F, X, A, alg: Algebra, h=1e-6, frame=None, r_frame=None):
    _F = derivative(lambda X: alg.sp(F(X), (A)), X, alg, h=h, frame=frame, r_frame=r_frame)
    return _F


def extract_first_value(obj):
    if isinstance(obj, list):
        return obj[0] if obj else None  # Return the first element if the list is not empty
    else:
        return obj  # Return None if the object is neither a list nor an integer


def extract_scalar(obj):
    if isinstance(obj, (MultiVector, list)):
        return obj[0]
    if isinstance(obj, (int, float)):
        return obj


def sort_grades_by_normsq(A):
    return sorted([(A.grade(g), g) for g in A.grades], key=lambda item: normsq(item[0]))


def blade_split(A, alg):
    Ar, g = sort_grades_by_normsq(A)[-1]
    projects = sorted([P(e, Ar) for e in alg.frame], key=normsq)[-g:]
    wed = gprod(projects)
    ratio = np.median(terms_ratio(wed, Ar))
    projects[0] *= ratio
    return projects


def vectors_partial(F, vectors, directions, h=1e-6):
    r = len(vectors)
    drF = 0
    for offset in product([1,-1], repeat=r):
        offset = np.array(offset)
        coef = np.prod(offset)
        offset = offset * h
        drF += 1/(2*h)**r * coef * F([a+v*d for a, v, d in zip(vectors, directions, offset)])
    return drF


def skew_symmetrizer(F, vectors, alg, h=1e-6):
    frame = alg.frame
    r_frame = reciprocal(alg.frame)
    drF = 0
    r = len(vectors)
    Ar = wedge(vectors)
    for base_vectors, reci_vectors in zip(permutations(frame, r), permutations(r_frame, r)):
        # if F is linear, vectors cause no difference in vectors_partial. The input is just Ar.
        drF += (Ar | wedge(base_vectors[::-1])) * vectors_partial(F, vectors, reci_vectors, h)
    return (1/factorial(r)) * drF


def simplicial_derivative(F, vectors, alg, h=1e-6, frame=None, r_frame=None):
    if not frame:
        frame = alg.frame
        r_frame = reciprocal(alg.frame)
    drF = 0
    r = len(vectors)
    for base_vectors, reci_vectors in zip(permutations(frame, r), permutations(r_frame, r)):
        drF += wedge(base_vectors[::-1]) * vectors_partial(F, vectors, reci_vectors, h=h)
    return (1/factorial(r)) * drF


def terms_ratio(A, B: MultiVector):
    valid_keys = [k for k in B.keys() if not np.isclose(B[k], 0)]
    return np.divide([A[k] for k in valid_keys], [B[k] for k in valid_keys])


def blade_exp(B) -> MultiVector:
    t = norm(B)
    b = B / t
    if (B ** 2)[0] > 0:
        return (np.cosh(t) + np.sinh(t) * b)
    else:
        return (np.cos(t) + np.sin(t) * b)


def blade_log(e):
    b = e.grade(2).normalized()
    t = np.arccos(e[0])
    return t * b
