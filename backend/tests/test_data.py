from hand_model.data import get_processed_paths


def test_processed_paths():
    paths = get_processed_paths()
    assert "images" in paths and "labels" in paths
