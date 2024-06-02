declare -a experiments=("chair"  "drums"  "ficus"  "hotdog"  "lego"  "materials"  "mic"  "ship")
# run command
for exp in "${experiments[@]}"
do
    python scripts/run.py --n_steps 100000 --mode nerf --nerf_compatibility --scene nerf_synthetic/$exp/transforms_train.json --test_transforms nerf_synthetic/$exp/transforms_test.json --screenshot_transforms nerf_synthetic/$exp/transforms_test.json --width 800 --height 800 --screenshot_dir test_nerf_synthetic/$exp/test
    python eval_metrics.py --gt_folder nerf_synthetic/$exp/test/ --pred_folder test_nerf_synthetic/$exp/test --csv_file test_nerf_synthetic/$exp/metrics.csv
done

