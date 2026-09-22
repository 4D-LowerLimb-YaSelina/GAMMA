#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

SCAN_DIR=/home/knpob/Documents/Hinton/data/shape-original/faust_test/scans
REGIST_DIR=/home/knpob/Documents/Hinton/data/shape-original/faust_test/registrations
TEXTURE_PATH=figures/texture-symbol-grid.png

SUFFIXES=(5k    10k    20k    40k    80k    120k)
COUNTS=(5000  10000  20000  40000  80000  120000)

for i in "${!SUFFIXES[@]}"; do
    SUFFIX=${SUFFIXES[$i]}
    COUNT=${COUNTS[$i]}
    OUT_DIR=output/faust_raw_${SUFFIX}
    LOG_FILE=${OUT_DIR}/run.txt

    mkdir -p "$OUT_DIR"

    echo "=== Decimation: ${SUFFIX} (${COUNT} vertices) ===" | tee "$LOG_FILE"

    # 1. Articulation
    conda activate sam3d
    python -m src.sam3d_grounding.run \
        --mesh_dir "$SCAN_DIR" \
        --out_dir "$OUT_DIR" \
        --no_trim_mesh \
        --decimation "$COUNT" \
        --device cuda \
        2>&1 | tee -a "$LOG_FILE"

    # Trim log to last 10 lines
    tail -n 10 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"

    # 2. Ground-truth correspondence
    conda activate fmap
    python src/faust_raw/export_s2t.py \
        --raw_mesh_folder "$OUT_DIR/decimated" \
        --raw_mesh_type obj \
        --regist_mesh_folder "$REGIST_DIR" \
        --export_folder "$OUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    # 3. Spectral preprocessing
    python src/faust_raw/preprocess.py \
        --raw_mesh_folder "$OUT_DIR/decimated" \
        --regist_mesh_folder "$REGIST_DIR" \
        --raw_mesh_type obj \
        --regist_mesh_type ply \
        --output_dir "$OUT_DIR" \
        --k_eig 200 \
        --plot_lbo \
        2>&1 | tee -a "$LOG_FILE"

    # 4. Benchmark
    python src/faust_raw/corr-bench.py \
        --data_root "$OUT_DIR" \
        --raw_mesh_folder "$OUT_DIR/decimated" \
        --regist_mesh_folder "$OUT_DIR/mhr" \
        --mesh_type obj \
        --methods spatial-nn spectral-nn spectral-icp \
        --num_evecs 200 \
        --n_iters 1 \
        --export_texture_transfer \
        --texture_transfer_dir "$OUT_DIR/texture-transfer" \
        --texture_path "$TEXTURE_PATH" \
        --device cuda \
        2>&1 | tee -a "$LOG_FILE"

    echo "=== Done: ${SUFFIX} ===" | tee -a "$LOG_FILE"
done
