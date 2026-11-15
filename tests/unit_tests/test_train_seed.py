"""train.py records each run's seed in its run folder."""

import json

from scripts.train import SEED_FILE, save_seed


def test_save_seed_writes_seed_and_its_source(tmp_path):
    run_dir = tmp_path / "logs" / "go2_lowfric" / "Nov15_00-00-00_"
    path = save_seed(str(run_dir), 4821, explicit=False)
    assert path == str(run_dir / SEED_FILE)
    assert json.loads((run_dir / SEED_FILE).read_text()) == {
        "seed": 4821,
        "source": "random",
    }
    save_seed(str(run_dir), 2, explicit=True)
    assert json.loads((run_dir / SEED_FILE).read_text())["source"] == "--seed"
