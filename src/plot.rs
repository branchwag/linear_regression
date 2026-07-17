//! Chart rendering for the linear regression results.
//!
//! Two charts: the Burn equivalent of the notebook's `plot_predictions` — the
//! training split, the held-out test split, and the model's predictions for
//! that split on one scatter — and the train/test loss curves over epochs.

use plotters::prelude::*;
use std::error::Error;

// Chart surface and ink.
const SURFACE: RGBColor = RGBColor(0xfc, 0xfc, 0xfb);
const TEXT_PRIMARY: RGBColor = RGBColor(0x0b, 0x0b, 0x0b);
const TEXT_SECONDARY: RGBColor = RGBColor(0x52, 0x51, 0x4e);
const GRID: RGBColor = RGBColor(0xe2, 0xe2, 0xdf);

// Categorical slots 1-3, in fixed order. Each series also carries a distinct
// marker shape: the hues alone sit in the colour-vision-deficiency floor band,
// so shape is what keeps the series apart.
const SERIES_TRAIN: RGBColor = RGBColor(0x2a, 0x78, 0xd6);
const SERIES_TEST: RGBColor = RGBColor(0x00, 0x83, 0x00);
const SERIES_PRED: RGBColor = RGBColor(0xe8, 0x7b, 0xa4);

const MARKER: i32 = 5;

/// Min and max of `values`, padded by 5% of the span so marks don't touch the
/// plot edge. Returns a unit-wide window if `values` is empty.
fn bounds(values: impl Iterator<Item = f32>) -> (f32, f32) {
    let (min, max) = values.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    let pad = ((max - min) * 0.05).max(0.05);
    (min - pad, max + pad)
}

/// Render the train/test loss curves to `path` as a PNG.
///
/// The y axis is logarithmic: MSE falls several orders of magnitude here, and a
/// linear axis would flatten everything after the first few epochs into the
/// baseline. Non-positive losses can't be drawn on a log axis, so values are
/// clamped up to the smallest positive loss in the run.
pub fn plot_loss_curves(
    train_loss: &[f32],
    test_loss: &[f32],
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let epochs = train_loss.len().max(test_loss.len());
    if epochs == 0 {
        return Err("no loss values to plot".into());
    }

    let finite = || {
        train_loss
            .iter()
            .chain(test_loss)
            .copied()
            .filter(|v| v.is_finite())
    };
    let floor = finite()
        .filter(|v| *v > 0.0)
        .fold(f32::INFINITY, f32::min);
    let floor = if floor.is_finite() { floor } else { 1e-8 };
    let ceiling = finite().fold(f32::NEG_INFINITY, f32::max).max(floor);

    // Log axes need multiplicative padding, not additive.
    let y_range = (floor / 2.0..ceiling * 2.0).log_scale();

    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&SURFACE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Training and test loss (MSE)",
            ("sans-serif", 22).into_font().color(&TEXT_PRIMARY),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(0f32..epochs as f32, y_range)?;

    chart
        .configure_mesh()
        .light_line_style(SURFACE)
        .bold_line_style(GRID)
        .axis_style(GRID)
        .label_style(("sans-serif", 13).into_font().color(&TEXT_SECONDARY))
        .x_desc("Epoch")
        .y_desc("Loss (log scale)")
        .draw()?;

    let curve = |values: &[f32]| -> Vec<(f32, f32)> {
        values
            .iter()
            .enumerate()
            .map(|(epoch, v)| (epoch as f32, v.max(floor)))
            .collect()
    };

    // Solid for train, dashed for test: the two curves overlap closely, so the
    // stroke pattern carries the distinction rather than hue alone.
    chart
        .draw_series(LineSeries::new(
            curve(train_loss),
            SERIES_TRAIN.stroke_width(2),
        ))?
        .label("Train loss")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], SERIES_TRAIN.stroke_width(2)));

    chart
        .draw_series(curve(test_loss).windows(2).enumerate().filter_map(
            |(i, pair)| match (i % 2 == 0, pair) {
                (true, [a, b]) => Some(PathElement::new(
                    vec![*a, *b],
                    SERIES_TEST.stroke_width(2),
                )),
                _ => None,
            },
        ))?
        .label("Test loss")
        .legend(|(x, y)| {
            PathElement::new([(x, y), (x + 8, y)], SERIES_TEST.stroke_width(2))
        });

    chart
        .configure_series_labels()
        .background_style(SURFACE)
        .border_style(GRID)
        .label_font(("sans-serif", 14).into_font().color(&TEXT_PRIMARY))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Render the scatter to `path` as a PNG.
///
/// Each slice holds `(x, y)` pairs: `train` and `test` are the ground-truth
/// splits, `preds` the model's output at the test inputs.
pub fn plot_predictions(
    train: &[(f32, f32)],
    test: &[(f32, f32)],
    preds: &[(f32, f32)],
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let all = || train.iter().chain(test).chain(preds);
    let (x_min, x_max) = bounds(all().map(|(x, _)| *x));
    let (y_min, y_max) = bounds(all().map(|(_, y)| *y));

    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&SURFACE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Linear regression: predictions vs. ground truth",
            ("sans-serif", 22).into_font().color(&TEXT_PRIMARY),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .light_line_style(SURFACE)
        .bold_line_style(GRID)
        .axis_style(GRID)
        .label_style(("sans-serif", 13).into_font().color(&TEXT_SECONDARY))
        .x_desc("X")
        .y_desc("y")
        .draw()?;

    // Ground-truth splits: filled circles.
    chart
        .draw_series(
            train
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), MARKER, SERIES_TRAIN.filled())),
        )?
        .label(format!("Training data ({})", train.len()))
        .legend(|(x, y)| Circle::new((x + 10, y), MARKER, SERIES_TRAIN.filled()));

    chart
        .draw_series(
            test.iter()
                .map(|(x, y)| Circle::new((*x, *y), MARKER, SERIES_TEST.filled())),
        )?
        .label(format!("Testing data ({})", test.len()))
        .legend(|(x, y)| Circle::new((x + 10, y), MARKER, SERIES_TEST.filled()));

    // Predictions: crosses, so they read as distinct from the test points they
    // sit on top of even where the two coincide.
    chart
        .draw_series(preds.iter().map(|(x, y)| {
            Cross::new((*x, *y), MARKER + 1, SERIES_PRED.stroke_width(2))
        }))?
        .label(format!("Predictions ({})", preds.len()))
        .legend(|(x, y)| Cross::new((x + 10, y), MARKER + 1, SERIES_PRED.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(SURFACE)
        .border_style(GRID)
        .label_font(("sans-serif", 14).into_font().color(&TEXT_PRIMARY))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    Ok(())
}
