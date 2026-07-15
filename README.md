# linear_regression

A minimal linear regression model built with the [Burn](https://burn.dev) deep
learning framework in Rust. It's a translation of the linear regression model
from the [Learn PyTorch workflow notebook](https://www.learnpytorch.io/01_pytorch_workflow/)
(`01_pytorch_workflow.ipynb`) into idiomatic Burn.

## What it does

Fits the line `y = weight * x + bias` (true `weight = 0.7`, `bias = 0.3`) from
50 synthetic points split 80/20 into train/test sets. A two-parameter module
learns `weights` and `bias` by gradient descent and converges to the true
parameters, hitting near-zero test loss.

```
Learned params -> weights: 0.7001, bias: 0.3000
True params    -> weights: 0.7000, bias: 0.3000
```

## How it works

| Concept | Choice |
|---|---|
| Backend | `Autodiff<NdArray>` (CPU with gradient tracking) |
| Model | `#[derive(Module)]` struct with `Param<Tensor<B, 1>>` weights and bias |
| Loss | Mean squared error (MSE) |
| Optimizer | SGD with Nesterov momentum (0.9), learning rate 0.5 |
| Epochs | 100 (converges in ~50) |
| Persistence | `NamedMpkFileRecorder` — saves, reloads, and verifies predictions match |

MSE's gradient tapers off as predictions approach the targets, and momentum
accelerates convergence on the unnormalized input, so the model reaches the
true parameters in relatively few epochs.

## Running

```sh
cargo run --release
```

This trains the model, prints train/test loss every 10 epochs, reports the
learned parameters and test predictions, then saves the trained model to
`models/01_burn_workflow_model_0.mpk` and confirms a reloaded copy produces
identical predictions.

## Project layout

```
src/main.rs    Model, training loop, save/load
Cargo.toml     Dependencies (burn 0.21, ndarray + autodiff)
models/        Saved model weights (generated, gitignored)
```
