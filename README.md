# LSTM Sleep Diary Project

This project predicts sleep efficiency from a short history of sleep diary entries using an LSTM-based regression model. It includes two model variants, a preprocessing pipeline that builds user-aware sequences, and evaluation scripts that produce per-user metrics and plots.

## What the project does

The dataset in [data/sleep_diary.csv](data/sleep_diary.csv) is turned into sequences of past days. The model learns to predict the next day’s `sleep_efficiency` from those sequences.

The project is organized around three stages:

1. preprocess the raw diary data,
2. train the LSTM model,
3. evaluate the model on held-out users and compare variants.

## Model architecture

The main model is in [src/model.py](src/model.py). It is a sequence-to-one regressor: it reads a short time series and outputs one numeric prediction.

### Input

Each sample has shape `[batch_size, sequence_length, input_size]`.

For this project:

- `sequence_length` is `LOOKBACK_DAYS` / `SEQUENCE_LENGTH` from [config.py](config.py)
- `input_size` is 6 features per day

The six input features are:

- `sleep_duration`
- `sleep_latency`
- `waso`
- `wakeup@night`
- `dayofweek`
- `bedtime_hour`

### Core blocks

The attention-based model has four main parts:

1. **LSTM encoder**
	- Processes the full sequence one day at a time.
	- Learns temporal patterns across the lookback window.
	- Uses `hidden_size`, `num_layers`, and `dropout` from [config.py](config.py).

2. **Optional user embedding**
	- If user IDs are available, each user gets an embedding vector.
	- That embedding is projected into the initial hidden and cell states of the LSTM.
	- This lets the model start from a different internal state for each user, which helps personalize the prediction.

3. **Temporal attention layer**
	- After the LSTM produces one hidden vector per timestep, a small feed-forward network scores each timestep.
	- The scores are converted to weights with `softmax`.
	- The model then takes a weighted sum of all hidden states to build one context vector.
	- In simple terms: the model learns which days in the lookback window matter most.

4. **Fully connected prediction head**
	- The context vector is passed through a small MLP.
	- The final output is one value: the predicted sleep efficiency.

### Why this design was chosen

- The LSTM captures short-term sequential patterns.
- Attention lets the model emphasize the most useful days in the sequence instead of treating all days equally.
- User embeddings help the model adapt to personal sleep behavior.
- The final dense head turns the learned sequence representation into a single regression output.

## No-attention baseline

The comparison model is in [src/model_without_att.py](src/model_without_att.py).

It uses the same LSTM and the same optional user embedding, but it skips the attention block. Instead, it uses the last LSTM output directly as the context vector before the final prediction head.

This baseline is useful because it shows whether attention actually improves performance.

## Data preprocessing pipeline

The preprocessing code is in [src/data_preprocessor.py](src/data_preprocessor.py).

### 1. Load the CSV

- The pipeline reads [data/sleep_diary.csv](data/sleep_diary.csv).
- The `date` column is parsed as a date so time-based features can be extracted.

### 2. Create features

- `dayofweek` is extracted from the date.
- `bedtime_hour` is extracted from the bedtime column.
- The model also uses `sleep_duration`, `sleep_latency`, `waso`, and the night-wakeup indicator.

### 3. Clean and sort

- Missing `waso` values are filled with `0`.
- Rows are sorted by `userId` and `date` so the sequence order is correct.

### 4. Normalize per user

- Each user’s input features are scaled independently with `MinMaxScaler`.
- This keeps users with larger raw ranges from dominating training.

### 5. Build sequences

- The preprocessor takes the previous `LOOKBACK_DAYS` days as one input sequence.
- The target is the next day’s `sleep_efficiency`.
- Example: if `LOOKBACK_DAYS = 7`, the model sees 7 days and predicts the 8th day.

### 6. Keep user mapping

- Each sequence stores a `user_idx`.
- This is used for the optional user embedding.
- It also makes user-level splitting possible.

### 7. Split by user

- Train, validation, and test splits are made by user, not by random row.
- That means the test set contains users not seen during training.
- This reduces leakage and makes the evaluation more realistic.

## Training

Training happens in [src/train_validate.py](src/train_validate.py) for the attention model and [src/train_validate_without_att.py](src/train_validate_without_att.py) for the baseline.

The training setup uses:

- mean squared error loss
- Adam optimizer
- learning rate scheduling on validation loss
- early stopping
- checkpoint saving for the best validation model

The model is trained on user-split data from the preprocessor, and the best checkpoint is saved to the `models/` folder.

## Evaluation and comparison

Evaluation happens in [src/test_model.py](src/test_model.py) and [src/test_model_without_att.py](src/test_model_without_att.py).

These scripts:

- load the saved checkpoint,
- run the model on the test users,
- compute per-user metrics,
- compute aggregated metrics over the full test set,
- save JSON result files,
- create per-user prediction plots.

The model comparison plot is created in [src/visualize_per_user.py](src/visualize_per_user.py). It compares the aggregated metrics from the attention and no-attention runs side by side.

## Main files

- [config.py](config.py) stores the experiment settings.
- [run.py](run.py) runs both training pipelines and both evaluation pipelines.
- [src/model.py](src/model.py) contains the attention-based model.
- [src/model_without_att.py](src/model_without_att.py) contains the baseline model.
- [src/data_preprocessor.py](src/data_preprocessor.py) prepares the data.
- [src/train_validate.py](src/train_validate.py) trains the attention model.
- [src/train_validate_without_att.py](src/train_validate_without_att.py) trains the baseline model.
- [src/test_model.py](src/test_model.py) evaluates the attention model.
- [src/test_model_without_att.py](src/test_model_without_att.py) evaluates the baseline model.
- [src/visualize_per_user.py](src/visualize_per_user.py) creates plots and comparison figures.

## Outputs

- `models/best_model.pt` stores the best attention-model checkpoint.
- `models/best_model_without_att.pt` stores the best no-attention checkpoint.
- `models/history/` stores training history files for the attention model.
- `models/history_without_att_*.json` stores training summaries for the baseline.
- `evaluation_results/evaluation_results.json` stores attention-model evaluation metrics.
- `evaluation_results/evaluation_results_without_att.json` stores baseline evaluation metrics.
- `evaluation_results/plots_per_user/` stores attention-model per-user plots.
- `evaluation_results_without_att/plots_per_user/` stores baseline per-user plots.
- `evaluation_results/model_comparison.png` stores the comparison chart.

## How to run

Train and evaluate both models with the unified runner:

```bash
python run.py
```

If you want to run them separately:

```bash
python src/train_validate.py
python src/test_model.py
python src/train_validate_without_att.py
python src/test_model_without_att.py
```

## Important configuration values

- `LOOKBACK_DAYS` / `SEQUENCE_LENGTH`: how many previous days the LSTM sees.
- `INPUT_SIZE`: number of features per day.
- `HIDDEN_SIZE`: size of the LSTM hidden state.
- `NUM_LAYERS`: number of stacked LSTM layers.
- `DROPOUT`: regularization rate.
- `USER_EMB_DIM`: size of the user embedding.
- `BATCH_SIZE`: batch size for training and evaluation.
- `LEARNING_RATE`: optimizer step size.
- `WEIGHT_DECAY`: L2 regularization strength.
- `MAX_EPOCHS`: maximum number of training epochs.
- `EARLY_STOPPING_PATIENCE`: how long training waits without improvement.
- `LR_SCHEDULER_PATIENCE`: how long before reducing the learning rate.
- `TRAIN_RATIO` and `VAL_RATIO`: user-level split proportions.
- `RANDOM_SEED`: ensures reproducible splits and training.

## Current weaknesses

This project works as a solid baseline, but it still has a few important limitations:

- The dataset appears to be relatively small, so the model may overfit and the results may not generalize well.
- The train/validation/test split is done by user, which is good for leakage control, but it can also make the test set harder if users differ a lot from each other.
- The model uses only a small set of engineered features, so it may miss useful context such as stress, activity, caffeine, medication, or irregular schedules.
- The attention mechanism is simple temporal attention, not a more advanced transformer-style self-attention block.
- User embeddings help personalization, but they only work for users seen during training and do not solve cold-start cases.
- The evaluation focuses on regression metrics such as RMSE, MAE, and R², so it does not directly explain why a prediction was wrong.

## Future work

Good next steps for this project would be:

1. **Improve the feature set**
	- Add more sleep-related and lifestyle features if they are available.
	- Try to capture missing context such as exercise, caffeine, stress, and bedtime regularity.

2. **Test stronger sequence models**
	- Compare the current attention model with scaled dot-product self-attention or a transformer-based model.
	- Run ablation studies to see whether the attention block actually improves performance.

3. **Use more robust evaluation**
	- Add cross-validation or repeated user-level splits.
	- Report confidence intervals or variance across runs so the comparison is more reliable.

4. **Improve personalization**
	- Explore user-specific fine-tuning or hierarchical modeling.
	- Handle new users better with cold-start strategies.

5. **Add interpretability**
	- Show which input features and which timesteps influence each prediction.
	- This would make the model easier to explain in a meeting or presentation.
