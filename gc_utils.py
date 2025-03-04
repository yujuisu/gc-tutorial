import numpy as np
from functools import reduce
from itertools import combinations, product, permutations
from kingdon import Algebra, MultiVector
from math import factorial


def random_vector(alg: Algebra):
    vec = np.random.random(alg.d)
    return alg.vector(vec)


def random_multivector(alg: Algebra):
    n = 2**alg.d
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


def random_r_vectors(r, alg):
    return [random_vector(alg) for _ in range(r)]


def cyclic_reorder(_v, k):
    return _v[k:] + _v[:k]


def cross(A, B):
    return (A * B - B * A) / 2


def normsq(A: MultiVector):
    return abs(A.sp(A.reverse())[0])


def norm(A):
    return np.sqrt(normsq(A))


def normalize(A, tol=1e-6):
    n = norm(A)
    assert n > tol, "zero norm"
    return A / n


def inv(A: MultiVector, tol=1e-6):
    Ar = A.reverse()
    n = A.sp(Ar)[0]
    assert abs(n) > tol, f"normsq {n}"
    return Ar / n


# def is_null_generator(gen):
#     for _ in gen:
#         return False  # If we retrieve any value, it's not null
#     return True


# def is_null(obj):
#     if isinstance(obj, list | tuple):
#         return len(obj) == 0  # Check if the list is empty
#     elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
#         return is_null_generator(obj)
#     else:
#         raise TypeError("Object is neither a list nor a generator.")


def wedge(vectors) -> MultiVector:
    return reduce(lambda a, b: a ^ b, vectors, 1)


def gprod(vectors) -> MultiVector:
    return reduce(lambda a, b: a * b, vectors, 1)


def random_r_blade(r, alg):
    return wedge(random_r_vectors(r, alg))


def assert_simple(A, tol=1e-8):
    Asquare = A**2
    if isinstance(A, (int, float)):
        return Asquare
    if Asquare[1:]:
        assert np.max(np.abs(Asquare[1:])) < tol, f"{Asquare[1:]}, not simple"
    return Asquare[0]


def assert_not_simple(A, tol=1e-6):
    Asquare = A**2
    assert len(Asquare.grades) > 1, "simple"
    if Asquare[1:]:
        assert np.max(np.abs(Asquare[1:])) > tol, "simple"
    return Asquare[0]


def P(X, A: MultiVector, tol=1e-6):  # projection of X onto a simple blade A
    alg = A.algebra
    Asquare = assert_simple(A)
    if abs(Asquare) < tol:
        return alg.ip(X, A) | A
    return (1 / Asquare) * (alg.ip(X, A) | A)


def max_grade(B, tol=1e-6):
    for r in B.grades[::-1]:
        Br = B.grade(r)
        if np.max(np.abs(Br[:])) > tol:
            return r, Br


def gram_schmidt(vectors):
    o_vecs = [vectors[0]]
    Ar = vectors[0]
    for v in vectors[1:]:
        Ar1 = Ar ^ v
        o_vecs.append(Ar | Ar1)
        Ar = Ar1
    return o_vecs


def reciprocal(blades):
    invI = inv(wedge(blades))
    dualblades = []
    for k in range(len(blades)):
        sign = (-1) ** (
            blades[k].grades[0] * sum(blade.grades[0] for blade in blades[:k])
        )
        dualblades.append(sign * wedge(blades[:k] + blades[k + 1 :]) * invI)
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


def difference(F, X, A, h=1e-6):
    d = (h) * A
    return F(X + d) - F(X - d)


def differential(F, X, A, h=1e-6):
    return (1 / (2 * h)) * difference(F, X, A, h)


def frame_gen(x, alg, **params):
    """Generate frame and reciprocal frame pairs based on provided parameters."""
    grade = params.get("grade")
    frame = params.get("frame")
    r_frame = params.get("r_frame")
    Ix = params.get("Ix")

    if not frame:
        if grade is not None:
            frame = r_vector_frame(alg.frame, grade)
            r_frame = r_vector_frame(reciprocal(alg.frame), grade, reverse=True)
        elif Ix:
            frame = blade_split(Ix(x), alg)
            r_frame = reciprocal(frame)
        else:
            frame = multi_frame(alg.frame)
            r_frame = reci_frame(alg.frame)

    return zip(frame, r_frame)


def derivative(F, X, alg: Algebra, h=1e-6, operator=None, **params):
    """Compute the derivative of a function F at X."""
    if not operator:
        operator = alg.gp

    frame_pairs = frame_gen(X, alg, **params)
    return sum(operator(r, differential(F, X, v, h)) for v, r in frame_pairs)


def derivative_gen(x, alg, h=1e-6, **params):
    """Generate derivatives in a lazy evaluation style."""
    for v, r in frame_gen(x, alg, **params):
        yield (r, lambda F: differential(F, x, v, h))


def curl(F, X, alg: Algebra, **params):
    """Compute the curl of a function F at X using the outer product."""
    return derivative(F, X, alg, operator=alg.op, **params)


def div(F, X, alg: Algebra, **params):
    """Compute the divergence of a function F at X using the inner product."""
    return derivative(F, X, alg, operator=inner, **params)


def adjoint(F, X, A, alg: Algebra, **params):
    _F = derivative(lambda X: alg.sp(F(X), (A)), X, alg, **params)
    return _F


def codiff(F, x, a, Ix, h=1e-6):
    return P(differential(F, x, P(a, Ix(x)), h), Ix(x))


def cocurl(A, x, alg, Ix, h=1e-6):
    return P(curl(A, x, alg, Ix=Ix, h=h), Ix(x))


def codiv(A, x, alg, Ix, h=1e-6):
    return P(div(A, x, alg, Ix=Ix, h=h), Ix(x))


def coderivative_gen(x, alg: Algebra, Ix, h=1e-6):
    for v, r in frame_gen(x, alg, Ix=Ix):
        yield (r, lambda F: codiff(F, x, v, Ix, h))


# def d(T, x, B, alg: Algebra, Ix, h=1e-6):
#     frame = blade_split(Ix(x), alg)
#     r_frame = reciprocal(frame)
#     return sum(
#         differential(lambda x: T(x, alg.ip(B, r)), x, v, h=h)
#         for v, r in zip(frame, r_frame)
#     )


# exterior differential
def d(T, x, B, alg, Ix, h=1e-6):
    return sum(
        o(lambda x: T(x, inner(B, r))) for r, o in derivative_gen(x, alg, h, Ix=Ix)
    )


def projection_differential(Ix, a, h=1e-3):
    return lambda A: lambda x: differential(lambda a: P(A(x), Ix(a)), x, P(a(x), Ix(x)), h=h)


def shape(Ix, alg):
    return lambda x, A: derivative(lambda x: P(A, Ix(x)), x, alg, Ix=Ix)


def extract_first_value(obj):
    if isinstance(obj, list):
        return (
            obj[0] if obj else None
        )  # Return the first element if the list is not empty
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
    ratio = np.median(terms_ratio(Ar, wed))
    projects[0] *= ratio
    return projects


def vectors_partial(F, vectors, directions, h=1e-6):
    r = len(vectors)
    offsets = np.array(list(product([1, -1], repeat=r)), dtype=np.float64)
    coefs = np.prod(offsets, axis=1)
    offsets *= h
    points = (
        [v + d * o for v, d, o in zip(vectors, directions, offset)]
        for offset in offsets
    )
    drF = sum(coef * F(point) for coef, point in zip(coefs, points))
    return 1 / (2 * h) ** r * drF


# def skew_symmetrizer(F, vectors, alg, h=1e-6):
#     frame = alg.frame
#     r_frame = reciprocal(alg.frame)
#     drF = 0
#     r = len(vectors)
#     Ar = wedge(vectors)
#     for base_vectors, reci_vectors in zip(permutations(frame, r), permutations(r_frame, r)):
#         # if F is linear, vectors cause no difference in vectors_partial. The input can be just Ar.
#         drF += (Ar | wedge(base_vectors[::-1])) * vectors_partial(F, vectors, reci_vectors, h)
#     return (1/factorial(r)) * drF


def skew_symmetrizer(F, Ar, alg, h=1e-6, frame=None, r_frame=None):
    if not frame:
        frame = alg.frame
        r_frame = reciprocal(alg.frame)
    drF = 0
    r = Ar.grades[0]
    for base_vectors, reci_vectors in zip(
        permutations(frame, r), permutations(r_frame, r)
    ):
        # if F is linear, vectors cause no difference in vectors_partial. The input can be just Ar.
        # So actually we allow Ar being any r-vector
        # I use np.zeros(r) to replace vectors
        drF += (Ar.sp(wedge(base_vectors[::-1]))) * vectors_partial(
            F, np.zeros(r), reci_vectors, h=h
        )
    return (1 / factorial(r)) * drF


def simplicial_derivative(F, vectors, alg, h=1e-6, frame=None, r_frame=None):
    if not frame:
        frame = alg.frame
        r_frame = reciprocal(alg.frame)
    drF = 0
    r = len(vectors)
    for base_vectors, reci_vectors in zip(
        permutations(frame, r), permutations(r_frame, r)
    ):
        drF += wedge(base_vectors[::-1]) * vectors_partial(
            F, vectors, reci_vectors, h=h
        )
    return (1 / factorial(r)) * drF


def terms_ratio(A, B: MultiVector):
    valid_keys = [k for k in B.keys() if not np.isclose(B[k], 0)]
    return np.divide([A[k] for k in valid_keys], [B[k] for k in valid_keys])


def blade_exp(B, tol=1e-6) -> MultiVector:
    signature = (B**2)[0]
    if abs(signature) < tol:
        return 1 + B
    t = norm(B)
    b = B / t
    if signature > tol:
        return np.cosh(t) + np.sinh(t) * b
    if signature < -tol:
        return np.cos(t) + np.sin(t) * b


def simple_rotor_log(R: MultiVector, tol=1e-6):
    blade = R.grade(2)
    signature = (blade**2)[0]
    if signature > tol:
        return np.arccosh(R[0]) * normalize(blade)
    if signature < -tol:
        return np.arccos(R[0]) * normalize(blade)
    else:
        return blade


def matrix2trans(M, alg: Algebra):
    d = alg.d
    assert M.shape[0] == d, "dimension not fit"
    return lambda x: sum(
        c * e for c, e in zip(np.dot(M, x.asfullmv()[1 : d + 1]), alg.frame)
    )


def simple_rotor_sqrt(R):
    # for rotations only, don't use on screw motions
    R_norm = norm(R + 1)
    assert R_norm >= 1e-4, "no explicit square root for -1"
    return (R + 1) / R_norm


def outermorphism(f, A: MultiVector, alg, h=1e-6, frame=None, r_frame=None):
    wf = lambda vectors: wedge([f(v) for v in vectors])
    outer = 0
    for r in A.grades:
        if r == 0:
            outer += A.grade(0)
            continue
        outer += skew_symmetrizer(wf, A.grade(r), alg, h, frame, r_frame)
    return outer


def outermorphism_(f, field, alg, h=1e-6, frame=None, r_frame=None, Ix=None):
    def f_outer(x):
        nonlocal frame, r_frame
        if Ix:
            frame = blade_split(Ix(x), alg)
            r_frame = reciprocal(frame)
        wf = lambda vectors: wedge([f(v)(x) for v in vectors])
        outer = 0
        A = field if isinstance(field, MultiVector) else field(x)
        for r in A.grades:
            if r == 0:
                outer += A.grade(0)
                continue
            outer += skew_symmetrizer(wf, A.grade(r), alg, h, frame, r_frame)
        return outer
    return f_outer


def adjoint_outermorphism(f, A, alg, h=1e-6, frame=None, r_frame=None):
    F = lambda vectors: wedge([f(v) for v in vectors]).sp(A)
    outer = 0
    for r in A.grades:
        if r == 0:
            outer += A.grade(0)
            continue
        # Why np.zeros(r)? The derivative of a linear function is constant
        outer += simplicial_derivative(F, np.zeros(r), alg, h, frame, r_frame)
    return outer


def adjoint_outermorphism_(f, field, alg, h=1e-6, frame=None, r_frame=None, Ix=None):
    def f_outer(x):
        nonlocal frame, r_frame
        if Ix:
            frame = blade_split(Ix(x), alg)
            r_frame = reciprocal(frame)
        F = lambda vectors: wedge([f(v)(x) for v in vectors]).sp(A)
        outer = 0
        A = field if isinstance(field, MultiVector) else field(x)
        for r in A.grades:
            if r == 0:
                outer += A.grade(0)
                continue
            # Why np.zeros(r)? The derivative of a linear function is constant
            outer += simplicial_derivative(F, np.zeros(r), alg, h, frame, r_frame)
        return outer
    return f_outer


def sym_part(f, alg):
    return lambda x: (f(x) + adjoint_outermorphism(f, x, alg)) / 2


def skew_part(f, alg):
    return lambda x: (f(x) - adjoint_outermorphism(f, x, alg)) / 2


def wedge_power(A, r):
    if r == 0:
        return 1
    result = A
    for _ in range(r - 1):
        result = A ^ result
    return result


def bivector_split(F, alg: Algebra):
    m = alg.d // 2
    if m <= 1:
        return F
    Ck = [(1 / factorial(r)) * wedge_power(F, r) for r in range(m + 1)]
    Ck2 = [(-1) ** (m - i) * alg.sp(Ck[i], Ck[i])[0] for i in range(m + 1)]
    Lk = np.roots(Ck2)
    mv_map_list = [{} for _ in range(m)]
    ck_inner_list = [alg.ip(Ck[i - 1], Ck[i]) for i in range(1, m + 1)]
    mv_keys = set.union(
        *(set(ck.keys()) for ck in ck_inner_list if isinstance(ck, MultiVector))
    )
    linear_eq_matrix = np.array(
        [[np.prod(Lk[i : i + k]) for i in range(m)] for k in range(m)]
    )
    inv_matrix = np.linalg.inv(linear_eq_matrix)
    for key in mv_keys:
        if all([np.isclose(ck[key], 0) for ck in ck_inner_list]):
            continue
        ans = np.dot(inv_matrix, [ck[key] for ck in ck_inner_list])
        for i, v in enumerate(ans):
            mv_map_list[i][key] = v
    return [alg.multivector(mv_map) for mv_map in mv_map_list]


def trans2matrix(f, alg: Algebra):
    values = [f(a) for a in alg.frame]
    return np.array([[ar.sp(v)[0] for v in values] for ar in reciprocal(alg.frame)])


def det(f, I, Ip: MultiVector, alg):
    return (outermorphism(f, I, alg) * Ip.reverse() / normsq(Ip))[0]


def random_versor(r, alg):
    return gprod(random_r_vectors(r, alg))


def check_isometry(f, alg, g=None, tol=1e-6):
    x, y = random_r_vectors(2, alg)
    if g:
        return max_diff(f(x) | g(f(y)), x | g(y)) < tol
    return max_diff(f(x) | f(y), x | y) < tol


# def check_symmetry(f, alg, tol=1e-6, grade=1):
#     x, y = (random_r_blade(grade, alg) for _ in range(2))
#     return max_diff(y | f(x), f(y) | x) < tol


def check_symmetry(F, alg: Algebra, tol=1e-6, grade=None):
    if grade:
        X = random_r_blade(grade, alg)
    else:
        X = random_multivector(alg)
    return max_diff(adjoint(F, 0, X, alg, grade=grade), F(X)) < tol


# def check_skew(f, alg, tol=1e-6, grade=1):
#     x, y = (random_r_blade(grade, alg) for _ in range(2))
#     return max_diff(y | f(x), -f(y) | x) < tol


def check_skew(F, alg: Algebra, tol=1e-6, grade=None):
    if grade:
        X = random_r_blade(grade, alg)
    else:
        X = random_multivector(alg)
    return max_diff(adjoint(F, 0, X, alg, grade=grade), -F(X)) < tol


# def lie_bracket(ax, bx, Ix=None, h=1e-6):
#     if Ix:
#         return lambda x: differential(bx, x, P(ax(x), Ix(x)), h=h) - differential(
#             ax, x, P(bx(x), Ix(x)), h=h
#         )
#     return lambda x: differential(bx, x, ax(x), h=h) - differential(ax, x, bx(x), h=h)


def lie_bracket(A, B, alg, Ix, h=1e-6):
    return lambda x: sum(
        (inner(A(x), r) ^ o(B)) - (o(A) ^ inner(r, B(x)))
        for r, o in derivative_gen(x, alg, h, Ix=Ix)
    )


# For fields on the manifold, the dual should be just on the manifold
# For fields in the embedding space, the dual should be in the embedding space
def dual_bracket(A, B, alg, nIx, h=1e-6):
    # nIx: normalized pseudoscalar
    return lambda x: lie_bracket(A, lambda x: B(x) * nIx(x), alg, nIx, h)(x) * inv(
        nIx(x)
    )


def dual_bracket_emb(A, B, alg: Algebra, Ix, h=1e-6):
    return lambda x: lie_bracket(A, lambda x: B(x).dual(), alg, Ix, h)(x).undual()


# 5.12c adjoint-outermorphism is the pullback/chain rule of derivatives
# without explicit define the inverse of f, we push forward the frame

def f_derivative_gen(x, f, alg, Ix, h=1e-6):
    f_vec = lambda a: lambda x: differential(f, x, a)
    f_ = lambda A: outermorphism_(f_vec, A, alg, Ix=Ix)
    for v, r in frame_gen(x, alg, Ix=f_(Ix)):
        yield (r, lambda F: differential(F, f(x), v, h))

# Say f: M -> N
# "back" as the derivative acts on a field G on N
# Yet mimicking the derivative of G(f(x)) on M


def back_derivative_gen(x, f, alg, Ix, h=1e-6):
    f_vec = lambda a: lambda x: differential(f, x, a)
    f_ = lambda A: outermorphism_(f_vec, A, alg, Ix=Ix)
    fa_ = lambda A: adjoint_outermorphism_(f_vec, A, alg, Ix=Ix)
    for v, r in frame_gen(x, alg, Ix=f_(Ix)):
        yield (fa_(r)(x), lambda F: differential(F, f(x), v, h))


# 5.14 "forward" as the derivative acts on a field F on M
# Yet mimicking the derivative of F(inv(f)(x)) on N

def forward_derivative_gen(x, f, alg, Ix, h=1e-6):
    f_vec = lambda a: lambda x: differential(f, x, a)
    f_ = lambda A: outermorphism_(f_vec, A, alg, Ix=Ix)
    nIx = lambda x: normalize(Ix(x))
    nI = nIx(x)
    iI1 = inv(f_(nIx)(x))
    for v, r in frame_gen(x, alg, Ix=Ix):
        yield iI1 * f_(nI*r)(x), lambda F: differential(F, x, v, h)