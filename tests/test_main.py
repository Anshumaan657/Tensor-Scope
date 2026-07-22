import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image

import main as application


def test_main_pipeline_runs(tmp_path, monkeypatch, capsys) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()

    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(
        image_directory / "first.jpg"
    )
    Image.new("RGB", (40, 30), color=(200, 150, 100)).save(
        image_directory / "second.png"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        application,
        "select_device",
        lambda: torch.device("cpu"),
    )

    application.main(show_visualization=False)

    output = capsys.readouterr().out
    assert "Shape: torch.Size([2, 3, 256, 256])" in output
    assert "Batch device: cpu" in output
    assert "Round trip preserved values: True" in output

    plt.close("all")
