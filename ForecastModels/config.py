import yaml
from types import SimpleNamespace


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(path="config.yaml"):
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = _to_namespace(raw)
    _validate(cfg)
    return cfg


def _validate(cfg):
    # SPLITS
    for name in ("train_set", "val_set", "test_set"):
        part = getattr(cfg, name)
        assert part.start < part.end, f"{name}: start must be before end"
    assert cfg.train_set.end < cfg.val_set.start < cfg.val_set.end < cfg.test_set.start, \
        "splits must be chronological and non-overlapping"
