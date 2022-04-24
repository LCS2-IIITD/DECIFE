# DECIFE: Detecting Collusive Users Involved in Blackmarket Following Services on Twitter

This is the pytorch implementation of DECIFE (In Proceedings of the 32nd ACM Conference on Hypertext and Social Media)

Please use the following citation is you use the work.

```
@inproceedings{dutta2021decife,
  title={DECIFE: Detecting collusive users involved in blackmarket following services on Twitter},
  author={Dutta, Hridoy Sankar and Aggarwal, Kartik and Chakraborty, Tanmoy},
  booktitle={Proceedings of the 32nd ACM Conference on Hypertext and Social Media},
  pages={91--100},
  year={2021}
}
```

### Installation:

1. `git clone https://github.com/LCS2-IIITD/DECIFE`

2. `cd DECIFE`

3. Create a virtual environment and run: `pip install -r requirements.txt`

4. Install [dgl v0.6.0](https://github.com/dmlc/dgl) from source.

### Data
The DGL heterogeneous graph along with the edge weights can be downloaded using this [google drive link](https://drive.google.com/file/d/1iakoPsbYLQWya8aLHPYKLggO-sW1EnVB/view?usp=sharing). Unzip the data folder into the root location.

### Usage

`python3 main.py --nu 0.2 --lr 0.6 --weight-decay 0.005 --n-epochs 100`

#### Parameters

```
--nu penalty parameter for hypershpere learning(0-1)
--lr learning rate
--weight-decay 
--n-epochs number of epochs for training
--seed
```
