# Deep Unlearning (MNIST reproduction)

This repo reuses the released code from **Deep Unlearning via Randomized Conditionally Independent Hessians (CVPR 2022)** with a few small tweaks to handle edge cases I hit while re-running the MNIST experiment. The focus here is showing exactly how to execute and replicate the run I used for my project.

## Prereqs
- Python environment with CUDA if you want GPU speedups.
- Install deps: `pip install -r reqs.txt`.

## How to run
1) Launch the MNIST retrain + scrubbing run:
```
bash ./scrub/scrub_scripts/mnist_retrain.sh
```
2) The script writes results to `scrub/results/reatrain_scrub_mnist_2nn.csv` (and related CSVs).
3) Plot the metrics after the run:
```
python scrub/results/plt.py
```

## Tuning hyperparameters
All knobs for the MNIST run live directly in `scrub/scrub_scripts/mnist_retrain.sh` (dataset/model choice, epochs, optimizer settings, number of removals, perturbations, etc.). Edit the variables at the top of that file to change defaults, then rerun the `bash` command above.

## Notes
- Code is derived from the original paper; only light modifications were made to cover edge cases encountered during reproduction.
- The scripts in `codec/` and `scrub/` remain as in the original release; the above commands are the intended entry points for this project write-up.
