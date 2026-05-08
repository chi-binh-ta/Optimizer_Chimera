def test_package_imports() -> None:
    import chimera

    assert chimera.Chimera21 is not None
    assert chimera.BitLinear is not None
    assert chimera.abs_stat is not None
