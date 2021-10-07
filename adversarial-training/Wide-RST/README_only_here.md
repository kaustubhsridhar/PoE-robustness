Wide-RST is an additional AT method. (Results on https://robustbench.github.io/ but not ACC paper)
It takes atleast 4 days to train one model and occupies both GPUs.

Instructions below:

| ${Approach} | Model     | lr (base) | lr (2/L) | lr (1/L) | ${File Name}           |
|:-----------:|-----------|:---------:|:--------:|:--------:|------------------------|
|   Wide-RST  | WRN-34-15 |    0.1    |  0.1485  |  0.0743  | Wide-RST_${lr}.pt      |

### 3.2.3 Wide-RST
#### 3.2.3.1 First download the unlabelled data used in RST from below into Wide-RST/data/ folder
[500K unlabeled data from TinyImages (with pseudo-labels)](https://drive.google.com/open?id=1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi)

#### 3.2.3.2 Then run the following commands
```
cd adversarial-training/Wide-RST/
python robust_self_training.py --lr ${lr} --aux_data_filename ti_500K_pseudo_labeled.pickle --model_dir Wide-RST_${lr}
```
* Run above command for three ${lr} as given in the table in section 2.2.2 of this README.
* Running the above command will generate checkpoints every 25 epochs for 200 epochs. At the end, lipschitz constant will be estimated.
* To only estimate Lispchitz constant (after training), run
```
python robust_self_training.py --lr ${lr} --model_dir RST_${lr} --only-lipschitz
```
* To evaluate on autoattack, run
```
python autoattack_evaluation.py --model_path Wide-RST_${lr}/checkpoint-epoch200.pt --log_file Wide-RST_${lr}/auto.log
```