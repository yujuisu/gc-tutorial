from kingdon import Algebra
import json

with open("./mesh_data/bunny_vertices.json", "r") as vf:
    vertices = json.load(vf)
with open("./mesh_data/bunny_faces.json", "r") as ff:
    f = json.load(ff)

alg = Algebra(3, 0, 1)
locals().update(alg.blades)


def half_edge(mesh):
    pts, faces = mesh
    HE = []

    # Process points: vector to pga point
    pts = [(e0 + alg.vector((0, *p)) - e2).dual() for p in pts]

    # Process faces: pga join
    faces = []
    for i, (v1, v2, v3) in enumerate(mesh[1]):
        face = {"vtx": [v1, v2, v3], "plane": pts[v1] & pts[v2] & pts[v3], "idx": i}
        HE.extend(
            [
                {"from": v1, "to": v2, "line": pts[v1] & pts[v2], "face": face},
                {"from": v2, "to": v3, "line": pts[v2] & pts[v3], "face": face},
                {"from": v3, "to": v1, "line": pts[v3] & pts[v1], "face": face},
            ]
        )

        face["edge"] = i * 3 + 2
        for j in range(3):
            HE[i * 3 + ((j + 2) % 3)]["next"] = HE[i * 3 + ((j + 1) % 3)]["prev"] = (
                i * 3 + j
            )

        faces.append(face)

    # An vertex gets assigned several times of half-edges
    vertex_list = [None] * len(pts)
    for idx, E in enumerate(HE):
        E["idx"] = idx
        vertex_list[E["from"]] = {"vtx": pts[E["from"]], "edge": idx}
    # Assign twin edges using a dictionary for efficient lookup
    edge_map = {}
    for idx, E in enumerate(HE):
        key = (E["from"], E["to"])
        edge_map[key] = idx

    for idx, E in enumerate(HE):
        twin_key = (E["to"], E["from"])
        twin_idx = edge_map.get(twin_key)
        E["twin"] = twin_idx

    return vertex_list, HE, faces


pts, HE, faces = half_edge((vertices, f))


def edges(p, f, s=None):
    idx = p["edge"]
    e = HE[idx]
    while True:
        s = f(e, s)
        e = HE[e["next"]] if p.get("plane") else HE[HE[e["twin"]]["next"]]
        if e["idx"] == p["edge"]:
            return s
