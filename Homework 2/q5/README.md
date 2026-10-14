# CS 521 HW2, Problem 5: ConstraintFlow starter code

## Files

| File | Content |
| --- | --- |
| `deeppoly.cf` | The DeepPoly certifier in ConstraintFlow. Use it with `provesound` and with `run_cf.py`. |
| `mnist_relu_3_100.onnx` | The MNIST network: 784-100-100-100-10, fully connected, ReLU. Test accuracy: 97.25%. |
| `run_cf.py` | Starter code. It compiles a certifier and runs it on the network. |
| `train_mnist.py` | The script that trained the network. You do not need to run it. |

The network takes a normalized image of shape (1, 1, 28, 28): `(x - 0.1307) / 0.3081`.
ConstraintFlow applies this normalization before the network.

## Setup

Use Python 3.10 or later.

```bash
python -m venv cfenv
source cfenv/bin/activate
git clone https://github.com/uiuc-focal-lab/constraintflow.git
cd constraintflow
git checkout 881e270
pip install "setuptools<81" -e .
cd ..
```

Do not use `pip install constraintflow` or `pip install git+https://...`. The PyPI version is old and does not work with `run_cf.py`. The `git+https` install does not include all the modules of ConstraintFlow.

`setuptools<81` is necessary because `z3` imports `pkg_resources`.
The starter code was tested with ConstraintFlow commit `881e270`.

## Check soundness

```bash
constraintflow provesound deeppoly.cf
```

Expected output: `Proved Affine` and `Proved Relu`.

## Run a certifier

```bash
python run_cf.py deeppoly.cf --num-images 100 --eps 0.005 0.01 0.02 0.03 0.05
```

The script downloads MNIST to `./data`.
For each epsilon, it gives the percentage of verified images and the time for each image.
An image is verified if the certifier proves that the true label has the largest logit for all inputs in the L-infinity ball.

Expected output for DeepPoly (first 100 test images, CPU):

| eps | verified |
| --- | --- |
| 0.005 | 99% |
| 0.01 | 94% |
| 0.02 | 83% |
| 0.03 | 49% |
| 0.05 | 2% |

### New shape members

The ConstraintFlow compiler initializes only the shape members `l`, `u`, `L`, `U` and `Z` at the input layer.
If your shape has a different member, use `--init` to initialize it as a copy of a known member.
For example, for a second polyhedral lower bound `La`:

```bash
python run_cf.py mycert.cf --init La=L
```

### Notes

- Use a different `.cf` file name for each certifier. The script compiles each certifier to `build_<name>/`.
- Run one certifier in each call of the script.
- The provesound examples in the ConstraintFlow repository use `true` as the stop condition. The compiler does not accept this. Use a named function, as in `deeppoly.cf`.
