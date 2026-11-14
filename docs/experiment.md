1. Nov 12

Modification:

- Add override of friction range in `train.py` and `play.py`. To be specific, input friction level (low, high) during training, which will be saved to an override json file. During playing, if not explicitly giving friction level, then it will use the friction config during training by default.

Experiment:

- Vanilla version of training on friction level: (0.1, 0.3)
The result is: it performs pretty well test time on friction level of 0.1, even OOD 0.05 level. But it performs bad on 1.0 high friction terrain.

- Vanilla version: training on (0.1, 1.0) road. But it performs bad on 1.0 high friction terrain.


2. Nov 13:
- Modification: update training config to 4.0 m/s velocities.