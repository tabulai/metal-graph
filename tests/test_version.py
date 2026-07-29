import metal_graph as mg


def test_python_and_native_versions_match():
    assert mg.__version__ == mg._core.version()
