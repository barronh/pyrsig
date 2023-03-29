def test_progress():
    from .. import _progress
    _progress(0, 1, 1)
    _progress(1, 1, 1)
    _progress(0, 1, 0)
    _progress(1, 1, 0)
