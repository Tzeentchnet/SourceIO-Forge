from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import pytest

from SourceIO.library.models.smd import (
    SMDBone,
    SMDDocument,
    SMDNode,
    SMDTriangle,
    SMDVertex,
    SMDVertexAnimationVertex,
    SMDWeight,
    dump,
    load,
    parse,
)


SKELETAL_SMD = """\
version 1
nodes
0 "root" -1
1 "child" 0
end
skeleton
time 1
0 1 2 3 0.1 0.2 0.3
1 4 5 6 0.4 0.5 0.6
time 0
0 0 0 0 0 0 0
1 0 1 0 0 0 0
end
triangles
models/example
0 0 0 0 0 0 1 0 0 2 0 0.25 1 0.75
1 1 0 0 0 0 1 1 0 0
0 0 1 0 0 0 1 0 1
end
"""


VTA = """\
version 1
nodes
0 "root" -1
end
skeleton
time 0
0 0 0 0 0 0 0
end
vertexanimation
time 1
2 1 2 3 0 0 1
time 0
0 0 0 0 0 0 1
1 1 0 0 0 1 0
end
"""


def _base_smd(extra: str = "") -> str:
    return (
        "version 1\n"
        "nodes\n"
        '0 "root" -1\n'
        '1 "child" 0\n'
        "end\n"
        "skeleton\n"
        "time 0\n"
        "0 0 0 0 0 0 0\n"
        "1 0 0 0 0 0 0\n"
        "end\n"
        f"{extra}"
    )


def _triangle_smd(vertex: str) -> str:
    fallback = "0 0 0 0 0 0 1 0 0"
    return _base_smd(
        "triangles\n"
        "material\n"
        f"{vertex}\n"
        f"{fallback}\n"
        f"{fallback}\n"
        "end\n"
    )


def _vertex(
    *,
    bone: int = 0,
    weights: tuple[SMDWeight, ...] | None = None,
) -> SMDVertex:
    return SMDVertex(
        bone,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0),
        weights,
    )


def test_parse_skeletal_smd_with_weighted_vertices():
    document = parse(SKELETAL_SMD)

    assert document.version == 1
    assert document.nodes == [
        SMDNode(0, "root", -1),
        SMDNode(1, "child", 0),
    ]
    assert document.skeleton[1][1] == SMDBone(
        1,
        (4.0, 5.0, 6.0),
        (0.4, 0.5, 0.6),
    )
    assert document.triangles[0].material == "models/example"
    assert document.triangles[0].vertices[0].weights == (
        SMDWeight(0, 0.25),
        SMDWeight(1, 0.75),
    )
    assert document.triangles[0].vertices[1].weights == ()
    assert document.triangles[0].vertices[2].weights is None


def test_parse_vta_vertexanimation():
    document = parse(VTA)

    assert document.triangles == []
    assert document.vertex_animation == {
        1: [
            SMDVertexAnimationVertex(
                2,
                (1.0, 2.0, 3.0),
                (0.0, 0.0, 1.0),
            )
        ],
        0: [
            SMDVertexAnimationVertex(
                0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            SMDVertexAnimationVertex(
                1,
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
        ],
    }


def test_comments_and_escaped_node_names_round_trip():
    text = r"""
// leading comment
version 1 // header comment
nodes // section comment
0 "root // retained" -1 // node comment
1 "say \"hello\" at C:\\rig" 0
end
skeleton
time 0 // frame comment
0 0 0 0 0 0 0
1 0 0 0 0 0 0
end // section terminator comment
"""

    document = parse(text)

    assert document.nodes[0].name == "root // retained"
    assert document.nodes[1].name == 'say "hello" at C:\\rig'
    serialized = dump(document)
    assert r'1 "say \"hello\" at C:\\rig" 0' in serialized
    assert parse(serialized) == document


def test_dump_is_deterministic_and_round_trips():
    document = parse(SKELETAL_SMD)

    first = dump(document)
    second = dump(parse(first))

    assert first == second
    assert first.index("time 0") < first.index("time 1")
    assert parse(first) == document
    assert dump(SMDDocument()) == (
        "version 1\n"
        "nodes\n"
        "end\n"
        "skeleton\n"
        "end\n"
    )


def test_dump_stream_and_load_path_or_caller_owned_stream(tmp_path: Path):
    document = parse(VTA)
    serialized = dump(document)
    output = StringIO()

    assert dump(document, output) is None
    assert output.getvalue() == serialized

    path = tmp_path / "face.vta"
    path.write_text(serialized, encoding="utf-8-sig")
    assert load(path) == document
    assert load(file=StringIO(serialized)) == document
    assert load(file=BytesIO(b"\xef\xbb\xbf" + serialized.encode())) == document
    with pytest.raises(ValueError):
        load()
    with pytest.raises(ValueError):
        load(path, StringIO(serialized))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "version 2\nnodes\nend\nskeleton\nend\n",
        "version 1\nnodes\nend\n",
        "version 1\nskeleton\nend\n",
        _base_smd("unknown\n"),
        _base_smd("nodes\nend\n"),
        'version 1\nnodes\n0 "root" -1\n',
        'version 1\nnodes\n0 "root" -1 trailing\nend\nskeleton\nend\n',
        'version 1\nnodes\n0 "bad\\q" -1\nend\nskeleton\nend\n',
        (
            "version 1\nnodes\n"
            '0 "root" -1\nend\n'
            "skeleton\n0 0 0 0 0 0 0\nend\n"
        ),
        (
            "version 1\nnodes\n"
            '0 "root" -1\nend\n'
            "skeleton\ntime 0\n0 0 0 0 0 0 0 trailing\nend\n"
        ),
        (
            "version 1\nnodes\n"
            '0 "root" -1\nend\n'
            "skeleton\ntime 0\ntime 0\nend\n"
        ),
        (
            "version 1\nnodes\n"
            '0 "root" -1\nend\n'
            "skeleton\ntime 0\n0 nan 0 0 0 0 0\nend\n"
        ),
        _triangle_smd("0 0 0 0 0 0 1 0 0 trailing"),
        _triangle_smd("0 0 0 0 0 0 1 0 0 2 0 1"),
        _base_smd(
            "vertexanimation\n"
            "0 0 0 0 0 0 1\n"
            "end\n"
        ),
        _base_smd(
            "vertexanimation\n"
            "time 0\n"
            "0 0 0 0 0 0 1 trailing\n"
            "end\n"
        ),
        _base_smd("trailing invalid tokens\n"),
    ],
    ids=[
        "empty",
        "version",
        "missing-skeleton",
        "missing-nodes",
        "unknown-section",
        "duplicate-section",
        "unterminated-section",
        "node-trailing-token",
        "node-invalid-escape",
        "pose-before-time",
        "pose-trailing-token",
        "duplicate-frame",
        "non-finite",
        "vertex-trailing-token",
        "weight-count",
        "vta-before-time",
        "vta-trailing-token",
        "trailing-invalid-tokens",
    ],
)
def test_parse_rejects_malformed_input(text: str):
    with pytest.raises(ValueError):
        parse(text)


@pytest.mark.parametrize(
    "text",
    [
        (
            "version 1\nnodes\n"
            '0 "root" 2\n'
            "end\nskeleton\nend\n"
        ),
        (
            "version 1\nnodes\n"
            '0 "a" 1\n'
            '1 "b" 0\n'
            "end\nskeleton\nend\n"
        ),
        (
            "version 1\nnodes\n"
            '0 "a" -1\n'
            '0 "b" -1\n'
            "end\nskeleton\nend\n"
        ),
        (
            "version 1\nnodes\n"
            '0 "root" -1\n'
            "end\nskeleton\ntime 0\n1 0 0 0 0 0 0\nend\n"
        ),
        _triangle_smd("2 0 0 0 0 0 1 0 0"),
        _triangle_smd("0 0 0 0 0 0 1 0 0 1 2 1"),
        (
            "version 1\nnodes\n"
            '0 "root" -1\n'
            "end\nskeleton\ntime 0\n"
            "0 0 0 0 0 0 0\n"
            "0 0 0 0 0 0 0\n"
            "end\n"
        ),
        _base_smd(
            "vertexanimation\n"
            "time 0\n"
            "0 0 0 0 0 0 1\n"
            "0 1 0 0 0 0 1\n"
            "end\n"
        ),
    ],
    ids=[
        "unknown-parent",
        "hierarchy-cycle",
        "duplicate-node",
        "unknown-pose-bone",
        "unknown-vertex-bone",
        "unknown-weight-bone",
        "duplicate-pose-bone",
        "duplicate-vta-index",
    ],
)
def test_parse_rejects_invalid_references_and_duplicates(text: str):
    with pytest.raises(ValueError):
        parse(text)


@pytest.mark.parametrize(
    "vertex",
    [
        "0 0 0 0 0 0 1 0 0 -1",
        "0 0 0 0 0 0 1 0 0 1 0 -0.1",
        "0 0 0 0 0 0 1 0 0 1 0 1.1",
        "0 0 0 0 0 0 1 0 0 2 0 0.6 1 0.5",
        "0 0 0 0 0 0 1 0 0 2 0 0.5 0 0.5",
        "0 0 0 0 0 0 1 0 0 1 0 inf",
    ],
    ids=[
        "negative-count",
        "negative",
        "greater-than-one",
        "sum-greater-than-one",
        "duplicate-bone",
        "non-finite",
    ],
)
def test_parse_rejects_invalid_weights(vertex: str):
    with pytest.raises(ValueError):
        parse(_triangle_smd(vertex))


@pytest.mark.parametrize(
    "document",
    [
        SMDDocument(version=2),
        SMDDocument(nodes=[SMDNode(0, "root", 0)]),
        SMDDocument(
            nodes=[SMDNode(0, "root", -1)],
            skeleton={
                0: [
                    SMDBone(
                        0,
                        (float("nan"), 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    )
                ]
            },
        ),
        SMDDocument(
            nodes=[SMDNode(0, "root", -1)],
            triangles=[
                SMDTriangle(
                    "material",
                    (
                        _vertex(weights=(SMDWeight(0, 0.75), SMDWeight(0, 0.25))),
                        _vertex(),
                        _vertex(),
                    ),
                )
            ],
        ),
        SMDDocument(
            nodes=[SMDNode(0, "root", -1)],
            triangles=[
                SMDTriangle("end", (_vertex(), _vertex(), _vertex()))
            ],
        ),
    ],
    ids=[
        "version",
        "hierarchy-cycle",
        "non-finite",
        "invalid-weights",
        "invalid-material",
    ],
)
def test_dump_rejects_invalid_documents(document: SMDDocument):
    with pytest.raises(ValueError):
        dump(document)
