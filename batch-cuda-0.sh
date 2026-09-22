# 5k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_5k/scans \
  out_dir=output/faust_5k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 10k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_10k/scans \
  out_dir=output/faust_10k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 20k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_20k/scans \
  out_dir=output/faust_20k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 40k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_40k/scans \
  out_dir=output/faust_40k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 80k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_80k/scans \
  out_dir=output/faust_80k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 120k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_120k/scans \
  out_dir=output/faust_120k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# raw
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_raw/scans \
  out_dir=output/faust_raw \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"
