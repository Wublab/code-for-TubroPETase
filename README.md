# code-for-TurboPETase

code for TurboPETase paper

Cui YL et al. Deep learning-aided redesign of a hydrolase for near 100% PET depolymerization under industrially relevant conditions. 2023. In submission.

## Install

```bash
conda create -n ml python=3.9
conda activate ml
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install x-transformers==0.31.2 
pip install mlm-pytorch
pip install numpy pandas scipy seaborn matplotlib
```

## Inference

see `mlm_data_analysis.ipynb`.

## Train

```bash
python mlm.py
```

## Note

The `*_new.pt` files have been updated to fit the newer version of `x-transformers`, BUT THEY HAVE NOT BEEN TESTED!  
WE STRONGLY RECOMMEND USING THE NOTEBOOK FOR REPRODUCING PURPOSES!

For scientific problems, please refer to wub_at_im.ac.cn or cuiyl_at_im.ac.cn (replace the _at_ with @ before sending the email).  
For code problems, please open a new issue.