# STEPS

## Raw FAUST dataset

### ULRSSM

```bash
cd utils/fmap

uv run train.py --config config/ulrssm-faust-raw/5k-xyz.yaml --device cuda:0
uv run train.py --config config/ulrssm-faust-raw/10k-xyz.yaml --device cuda:1
uv run train.py --config config/ulrssm-faust-raw/20k-xyz.yaml --device cuda:2

uv run train.py --config config/ulrssm-faust-raw/5k-wks.yaml --device cuda:3
uv run train.py --config config/ulrssm-faust-raw/10k-wks.yaml --device cuda:4
uv run train.py --config config/ulrssm-faust-raw/20k-wks.yaml --device cuda:5
```

Texture transfer:

```bash
uv run bench.py --folder experiment/ulrssm/5k-xyz-20260727-111013 --device cuda:1
uv run bench.py --folder experiment/ulrssm/5k-wks-20260727-111029 --device cuda:1
uv run bench.py --folder experiment/ulrssm/10k-xyz-20260727-111019 --device cuda:1
uv run bench.py --folder experiment/ulrssm/10k-wks-20260727-111031 --device cuda:1
uv run bench.py --folder experiment/ulrssm/20k-xyz-20260727-111022 --device cuda:1
uv run bench.py --folder experiment/ulrssm/20k-wks-20260727-111034 --device cuda:1
```

## BeCoS processed FAUST/SCAPE/KIDS dataset

### ULRSSM

FAUST-only partial-to-full and partial-to-partial:

```bash
cd utils/fmap

uv run train.py --config config/ulrssm-becos/faust_p2f_wks.yaml --device cuda:0
uv run train.py --config config/ulrssm-becos/faust_p2f_xyz.yaml --device cuda:0
uv run train.py --config config/ulrssm-becos/faust_p2p_wks.yaml --device cuda:1
uv run train.py --config config/ulrssm-becos/faust_p2p_xyz.yaml --device cuda:1
```

KIDS-only partial-to-full and partial-to-partial:

```bash
uv run train.py --config config/ulrssm-becos/kids_p2f_wks.yaml --device cuda:0
uv run train.py --config config/ulrssm-becos/kids_p2f_xyz.yaml --device cuda:0
uv run train.py --config config/ulrssm-becos/kids_p2p_wks.yaml --device cuda:1
uv run train.py --config config/ulrssm-becos/kids_p2p_xyz.yaml --device cuda:1
```

