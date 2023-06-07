# code-for-TurboPETase

code for TurboPETase paper

Cui YL et al. Deep learning-aided redesign of a hydrolase for near 100% PET depolymerization under industrially relevant conditions. 2023. In submission.

## Install

```bash
conda create -n ml python=3.9
conda activate ml
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install x-transformers
pip install mlm-pytorch
pip install numpy pandas scipy seaborn matplotlib tqdm
```

## Inference

see `mlm_data_analysis.ipynb`.

## Train

```bash
python mlm.py
```
We also provide a [colab notebook](https://colab.research.google.com/drive/1e4sgLM8QkJNhqwdotV38KnGS2eSJAolP?usp=sharing) to train on your own sequence. It utilized the colabfold's MSA.

## Note

The `*_new.pt` files have been updated to fit the newer version of `x-transformers`, BUT THEY HAVE NOT BEEN TESTED!  

For scientific problems, please refer to wub_at_im.ac.cn or cuiyl_at_im.ac.cn (replace the _at_ with @ before sending the email).  
For code problems, please open a new issue.