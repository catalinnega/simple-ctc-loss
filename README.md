# Simple implementation of the Connectionist Temporal Classification loss
Implementation as described in the paper: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks", Graves et al.: https://dl.acm.org/doi/abs/10.1145/1143844.1143891

Should be used for experimentation puposes.

## Install requirements
```
pip install setuptools
pip install -e .
```

### Run CTC loss test:
```
python3 ctc/run_test.py
```
### Inputs info:
```
Results summary:

Acoustic emissions matrix shape:
        torch.Size([125, 36])

Label to ID dictionary:
        {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'A': 6, 'I': 7, 'N': 8, 'U': 9, 'T': 10, 'R': 11, 'C': 12, 'O': 13, 'Ă': 14, 'S': 15, 'M': 16, 'D': 17, 'L': 18, 'P': 19, 'V': 20, 'Ș': 21, 'Ț': 22, 'F': 23, 'Î': 24, 'B': 25, 'Z': 26, 'G': 27, 'Â': 28, 'H': 29, 'X': 30, 'K': 31, 'J': 32, 'Y': 33, 'W': 34, 'Q': 35}

Transcript text:
        ÎȚI MULȚUMESC MULT

Transcript labels:
        ['<s>', 'Î', '<s>', 'Ț', '<s>', 'I', '<s>', '|', '<s>', 'M', '<s>', 'U', '<s>', 'L', '<s>', 'Ț', '<s>', 'U', '<s>', 'M', '<s>', 'E', '<s>', 'S', '<s>', 'C', '<s>', '|', '<s>', 'M', '<s>', 'U', '<s>', 'L', '<s>', 'T', '<s>', '|', '<s>']
Viterbi (argmax) decoding labels:
        ['<s>', 'Î', '<s>', 'Ț', '<s>', 'I', '<s>', '|', '<s>', 'M', '<s>', 'U', '<s>', 'L', '<s>', 'Ț', '<s>', 'U', '<s>', 'M', '<s>', 'E', '<s>', 'S', 'C', '<s>', '|', '<s>', 'M', '<s>', 'U', '<s>', 'L', '<s>', 'T', '<s>', '|', '<s>']
```

### Output results:
| Case name                                  |   Forward Viterbi negative loglikelihood |   Backwrd Viterbi negative loglikelihood |   Torch CTC loss |   Implemented CTC Loss |   Gradient matrix norm |   Joint probability norm |   Emissions Gradient norm |
|:-------------------------------------------|-----------------------------------------:|-----------------------------------------:|-----------------:|-----------------------:|-----------------------:|-------------------------:|--------------------------:|
| Half of IDs are wrong                      |                               -1343.07   |                               -1132.36   |        1006.6    |              1343.07   |              nan       |              1.41227     |                 nan       |
| No blank separators                        |                                 -44.3633 |                                 -44.3633 |          44.3633 |                44.3633 |                2.54267 |              1.40262e+07 |                   1.86154 |
| Transcript IDs                             |                                  22.9122 |                                  22.9123 |         -22.9123 |               -22.9122 |                1.52612 |              5.66188     |                   1.5262  |
| Transcript IDs with max probabilities == 1 |                                 110.704  |                                 110.704  |        -110.704  |              -110.704  |               36.2394  |              0.573886    |                  36.2393  |
| All IDs are the same                       |                                 nan      |                                 nan      |         986.052  |               nan      |              nan       |            nan           |                 nan       |

## References:
- "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks", Graves et al.: https://dl.acm.org/doi/abs/10.1145/1143844.1143891
- "Sequence Modeling With CTC", Hannun et al.: https://distill.pub/2017/ctc/
- CTC implementations:
    - https://github.com/amaas/stanford-ctc
    - https://github.com/vadimkantorov/ctc
