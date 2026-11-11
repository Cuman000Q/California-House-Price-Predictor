# California Housing Price Predictor

A neural network built with **PyTorch** that predicts the median house value of a
California district from 8 facts about it, such as median income, house age, number of
rooms and location.

**Result: R² = 0.73 on districts the model has never seen.** The model explains about
73% of the variation in house prices.

## Dataset

The [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
from the 1990 U.S. census, loaded through scikit-learn:

- 20,640 districts (block groups)
- 8 features: median income, house age, average rooms, average bedrooms, population,
  average occupancy, latitude, longitude
- Target: median house value, in units of **$100,000**

## How it works

1. **Load and split**: 80% of the districts for training, 20% for testing.
2. **Scale**: `StandardScaler`, fitted on the training data only.
3. **Batch**: the data becomes PyTorch tensors, fed in batches of 32 with `DataLoader`.
4. **Model**: a fully connected network with a single output neuron for the price:

   ```
   8 features → Linear(8, 30) → ReLU → Linear(30, 15) → ReLU → Linear(15, 1) → price
   ```

5. **Train**: 30 epochs with the Adam optimizer (learning rate 0.01) and mean squared
   error loss.
6. **Evaluate**: mean squared error (MSE) and R² on the test set.

## Results

```
========================================
=== MEAN SQUARED ERROR (MSE): 0.3506 ===
=== R² SCORE:                 0.7324 ===
========================================
```

- **MSE 0.35**: the square root (≈ 0.59) means a typical prediction is off by about
  **$59,000**.
- **R² 0.73**: 1.0 would be a perfect prediction; 0 would be no better than always
  guessing the average price.

The random seed is fixed, so you get the same result on every run.

## Getting started

Requires Python 3.9+.

```bash
git clone https://github.com/Cuman000Q/california-housing-price-predictor.git
cd california-housing-price-predictor
pip install -r requirements.txt
python housing_price_predictor.py
```

On the first run scikit-learn downloads the dataset (about 400 KB), so an internet
connection is needed once. Training takes under a minute on a normal CPU.

## Built with

- [PyTorch](https://pytorch.org/): the neural network and training loop
- [scikit-learn](https://scikit-learn.org/): dataset, train/test split, scaling and metrics

## Ideas for next steps

- [ ] Compare with a linear regression and a gradient-boosting baseline
- [ ] Hold out a validation set and stop training when it stops improving
- [ ] Plot predicted vs. real prices

## Author

Made by [Cuman000Q](https://github.com/Cuman000Q). See also the
[Wine Origin Classifier](https://github.com/Cuman000Q/wine-origin-classifier),
the classification counterpart of this project.
