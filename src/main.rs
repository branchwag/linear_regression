//! Linear regression in Burn, translated from the PyTorch workflow notebook.
//!
//! Fits `y = weight * x + bias` (true weight = 0.7, bias = 0.3) using MSE loss
//! and stochastic gradient descent, based on the PyTorch `nn.Module` version in
//! `01_pytorch_workflow.ipynb`.

mod plot;

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, Param};
use burn::optim::momentum::MomentumConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use burn::tensor::cast::ToElement;
use burn::tensor::{Distribution, Tensor, TensorData};
use std::error::Error;

/// The learnable linear regression model: `y = weights * x + bias`.
///
/// This is the Burn equivalent of the notebook's `LinearRegressionModel`
/// subclass of `nn.Module`, with two scalar parameters registered for autodiff.
#[derive(Module, Debug)]
struct LinearRegressionModel<B: Backend> {
    weights: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> LinearRegressionModel<B> {
    /// Initialize the parameters from a standard normal distribution, like
    /// PyTorch's `nn.Parameter(torch.randn(1))`.
    fn new(device: &B::Device) -> Self {
        let weights = Tensor::random([1], Distribution::Normal(0.0, 1.0), device);
        let bias = Tensor::random([1], Distribution::Normal(0.0, 1.0), device);
        Self {
            weights: Param::from_tensor(weights),
            bias: Param::from_tensor(bias),
        }
    }

    /// Forward pass: the linear regression formula `weights * x + bias`.
    ///
    /// `x` has shape `[n, 1]`; the `[1]` parameters broadcast across the batch.
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let weights = self.weights.val().reshape([1, 1]);
        let bias = self.bias.val().reshape([1, 1]);
        x.mul(weights).add(bias)
    }
}

/// Mean squared error (MSE loss). Its gradient shrinks as predictions approach
/// the targets, so training converges quickly and settles precisely.
fn mse_loss<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    (pred - target).powf_scalar(2.0).mean()
}

/// Build a `[n, 1]` tensor from a slice of `f32` values.
fn column<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values.to_vec(), [n, 1]), device)
}

/// Extract the scalar weight and bias values for logging.
fn params<B: Backend>(model: &LinearRegressionModel<B>) -> (f32, f32) {
    let w = model.weights.val().into_scalar().to_f32();
    let b = model.bias.val().into_scalar().to_f32();
    (w, b)
}

/// Render the per-epoch loss history to a chart. Returns the path written.
fn chart_loss(train: &[f32], test: &[f32]) -> Result<&'static str, Box<dyn Error>> {
    const PATH: &str = "plots/loss.png";
    std::fs::create_dir_all("plots")?;
    plot::plot_loss_curves(train, test, PATH)?;
    Ok(PATH)
}

/// Pair the raw inputs with the ground-truth splits and the model's test
/// predictions, then render them to a chart. Returns the path written.
fn chart_predictions<B: Backend>(
    xs: &[f32],
    ys: &[f32],
    split: usize,
    preds: &Tensor<B, 2>,
) -> Result<&'static str, Box<dyn Error>> {
    fn pair(xs: &[f32], ys: &[f32]) -> Vec<(f32, f32)> {
        xs.iter().copied().zip(ys.iter().copied()).collect()
    }

    let data = preds.to_data();
    let pred_ys = data
        .as_slice::<f32>()
        .map_err(|e| format!("prediction tensor is not f32: {e:?}"))?;

    const PATH: &str = "plots/predictions.png";
    std::fs::create_dir_all("plots")?;
    plot::plot_predictions(
        &pair(&xs[..split], &ys[..split]),
        &pair(&xs[split..], &ys[split..]),
        &pair(&xs[split..], pred_ys),
        PATH,
    )?;
    Ok(PATH)
}

fn main() {
    // Backend: ndarray with autodiff wrapped around it for gradient tracking.
    type B = Autodiff<NdArray>;
    let device = Default::default();

    // Reproducible parameter initialization (cf. torch.manual_seed(42)).
    // Note: RNG algorithms differ, so initial values won't match PyTorch exactly.
    <B as Backend>::seed(&device, 42);

    // 1. Prepping and loading data: y = bias + weight * X
    let weight = 0.7_f32;
    let bias = 0.3_f32;
    let xs: Vec<f32> = (0..50).map(|i| i as f32 * 0.02).collect();
    let ys: Vec<f32> = xs.iter().map(|x| bias + weight * x).collect();

    // Split 80/20 into train/test sets.
    let split = (0.8 * xs.len() as f32) as usize;
    let x_train = column::<B>(&xs[..split], &device);
    let y_train = column::<B>(&ys[..split], &device);
    let x_test = column::<B>(&xs[split..], &device);
    let y_test = column::<B>(&ys[split..], &device);

    // 2. Build the model.
    let mut model = LinearRegressionModel::<B>::new(&device);
    println!("Initial params: {:?}", params(&model));

    // 3. Train: MSE loss + SGD optimizer.
    //
    // MSE's gradient scales with the residual, so steps taper off automatically
    // as predictions approach the targets. The input isn't normalized, which
    // bounds how large a stable learning rate can be, so we add Nesterov
    // momentum to accelerate convergence onto the true parameters
    // (weight = 0.7, bias = 0.3) in relatively few epochs.
    let momentum = MomentumConfig {
        momentum: 0.9,
        dampening: 0.0,
        nesterov: true,
    };
    let mut optimizer = SgdConfig::new().with_momentum(Some(momentum)).init();
    let lr = 0.5;
    let epochs = 100;

    // Per-epoch loss history, kept for the loss-curve chart.
    let mut train_history = Vec::with_capacity(epochs);
    let mut test_history = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        // Forward pass + loss.
        let pred = model.forward(x_train.clone());
        let loss = mse_loss(pred, y_train.clone());

        // Backprop + gradient descent step. `backward` borrows, so the loss
        // tensor is still ours to read afterwards.
        let grads = loss.backward();
        let train_loss = loss.into_scalar().to_f32();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(lr, model, grads);

        // Testing: evaluate on the held-out set (no grad step taken).
        let test_pred = model.forward(x_test.clone());
        let test_loss = mse_loss(test_pred, y_test.clone()).into_scalar().to_f32();

        train_history.push(train_loss);
        test_history.push(test_loss);

        if epoch % 10 == 0 {
            let (w, b) = params(&model);
            println!(
                "Epoch: {epoch:>3} | Train loss: {train_loss:.5} | Test loss: {test_loss:.5} | weights: {w:.4}, bias: {b:.4}"
            );
        }
    }

    // Final learned parameters (true values: weight = 0.7, bias = 0.3).
    let (w, b) = params(&model);
    println!("\nLearned params -> weights: {w:.4}, bias: {b:.4}");
    println!("True params    -> weights: {weight:.4}, bias: {bias:.4}");

    // Predictions on the test set, charted against the ground-truth splits.
    let preds = model.forward(x_test.clone());
    match chart_predictions(&xs, &ys, split, &preds) {
        Ok(path) => println!("\nWrote predictions chart to {path}"),
        Err(e) => eprintln!("\nFailed to render predictions chart: {e}"),
    }
    match chart_loss(&train_history, &test_history) {
        Ok(path) => println!("Wrote loss curve chart to {path}"),
        Err(e) => eprintln!("Failed to render loss curve chart: {e}"),
    }

    // 4. Save the model (cf. torch.save(model.state_dict(), ...)).
    std::fs::create_dir_all("models").expect("failed to create models dir");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file("models/01_burn_workflow_model_0", &recorder)
        .expect("failed to save model");
    println!("\nSaved model to models/01_burn_workflow_model_0.mpk");

    // Load it back and confirm predictions match.
    let loaded = LinearRegressionModel::<B>::new(&device)
        .load_file("models/01_burn_workflow_model_0", &recorder, &device)
        .expect("failed to load model");
    let loaded_preds = loaded.forward(x_test);
    let original = preds.into_data();
    let reloaded = loaded_preds.into_data();
    let matches = original
        .as_slice::<f32>()
        .expect("prediction tensor is not f32")
        .iter()
        .zip(reloaded.as_slice::<f32>().expect("prediction tensor is not f32"))
        .all(|(a, b)| (a - b).abs() < 1e-6);
    println!("Loaded model predictions match original: {matches}");
}
