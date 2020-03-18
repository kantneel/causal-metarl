# causal-metarl
https://arxiv.org/pdf/1901.08162.pdf

Notes: 
- This repo uses stable-baselines to train policies. Unfortunately, the A2C implementation does not allow for us to specify a value for the momentum parameter in the RMSPropOptimizer class. Currently, I've opened a PR in the repo to address this, but you may find it easier to simply modify this [line of code](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/a2c/a2c.py#L182) accordingly. 
