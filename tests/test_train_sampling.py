import numpy as np

from decomp_judge.train_sampling import functional_text, select_candidates


def test_styling_is_removed_but_functional_settings_remain() -> None:
    first = "Please implement a forum. Users set privacy settings. Use ivory for the background."
    second = "Please implement a forum. Users set privacy settings. Apply red for the background."
    assert functional_text(first) == functional_text(second)
    assert "privacy settings" in functional_text(first)


def test_sampling_excludes_annotations_and_global_near_duplicates() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    chosen = select_candidates(vectors, [0, 1, 2, 3], 3, {3}, 0.9, [0])
    assert chosen == [2]


def test_sampling_handles_cluster_with_no_eligible_candidates() -> None:
    assert select_candidates(np.array([[1.0, 0.0]]), [0], 3, {0}, 0.9, []) == []
