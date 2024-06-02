declare -a experiments=("bicycle"  "bonsai"  "counter"  "flowers"  "garden"  "kitchen"  "room"  "stump"  "treehill")
# run command
for exp in "${experiments[@]}"
do
    python scripts/run.py --n_steps 100000 --mode nerf --nerf_compatibility --scene ../mipnerf_360/$exp/transforms_train.json --test_transforms ../mipnerf_360/$exp/transforms_test.json --screenshot_transforms ../mipnerf_360/$exp/transforms_test.json --screenshot_dir test_mipnerf_360/$exp/test
done

