# rnd

Random network distillation on the next observation: the bonus of a state is half the squared
distance between a fixed random target network's features of that state and a trained predictor's.

| file | what is in it |
|---|---|
| `config.py` | `RNDConfig` — the two network widths |
| `networks.py` | the target and predictor initialisation, their joint forward pass, and the input whitening |
| `implementation.py` | the family's hooks: initialise, warm up the statistics, score a rollout, contribute the predictor's error to the loss |

The predictor is trained by the agent's own optimizer: the composer builds one trainable tree
`{agent, bonus}` and one loss `agent + bonus`, so one Adam and one gradient-norm clip span both.
That is what the frozen 09_parallelization baseline did, and the golden-parity gate holds the
platform to it bit for bit.

The statistics are this family's own state, not the agent's: the whitening exists because the
predictor's input has to keep a stable scale as the agent moves through the maze, and a family
that scores from a table needs nothing of the kind.
