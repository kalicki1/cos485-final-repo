### How to stitch audio

First, downsample the baseline noisy and clean audio: 

```python data_downsample.py```

Then, stitch the downsampled baseline and noisy audio:

```python stitch_audio.py -s ../results/baseline_noisy_sliced/ -t ../results/baseline_noisy/```
```python stitch_audio.py -s ../results/baseline_clean_sliced/ -t ../results/baseline_clean/```


Create the Wiener filter baseline over the baseline noisy audio: 

```python data_wiener.py --filter-window 2 -s ../results/baseline_noisy/ -t ../results/baseline_wiener_2/```


Run composite evaluations of noisy data on clean source data:

```python eval_noisy_performance.py --test_wavs '../results/baseline_wiener_2/' --clean_wavs '../results/baseline_clean/' --logfile '../results/metrics_all.txt'```


My evaluations: 

```python evaluation.py --clean_folder '../results/baseline_clean/' --test_folder '../results/baseline_wiener_2' --log-file '../results/metrics_wiener_2.txt'```

