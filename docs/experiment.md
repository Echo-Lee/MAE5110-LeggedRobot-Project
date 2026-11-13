1. Nov 12

Modification:

- Add override of friction range in `train.py` and `play.py`. To be specific, input friction level (low, high) during training, which will be saved to an override json file. During playing, if not explicitly giving friction level, then it will use the friction config during training by default.

Experiment:

- Vanilla version of training on friction level: (0.1, 0.2)