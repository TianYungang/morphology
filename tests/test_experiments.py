import morphological


def test_main_runs_binary_then_grayscale_experiments(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        morphological,
        "run_binary_experiments",
        lambda: calls.append("binary"),
    )
    monkeypatch.setattr(
        morphological,
        "run_grayscale_experiments",
        lambda: calls.append("grayscale"),
    )

    morphological.main()

    assert calls == ["binary", "grayscale"]
