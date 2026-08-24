# NBA MVP Predictor

A machine learning project that predicts NBA Most Valuable Player (MVP) candidates from a player's season box-score stats. It trains binary classifiers on 22 years of past MVP winners versus a sample of non-MVP players, uses SMOTE to correct for the resulting class imbalance, and then applies the trained model to a full season of player stats to surface likely MVP-caliber performances.

## How it works

1. **Label the training data.** Every MVP winner from 2001–2022 (`NBAMVPdataset.csv`) is labeled `Yes`. A random sample of players from a full season's stats (`NBAPlayerStatsRegular.csv`) is labeled `No`.
2. **Features.** Each player is represented by 10 box-score stats: games played (`G`), minutes played (`MP`), points (`PTS`), rebounds (`TRB`), assists (`AST`), steals (`STL`), blocks (`BLK`), field goal % (`FG%`), 3-point % (`3P%`), and free throw % (`FT%`).
3. **Handle class imbalance.** Because MVP winners are rare compared to the full player pool, the notebook tries two approaches: downsampling the non-MVP class to match the 22 MVP examples, and, in a second pass, oversampling the minority (MVP) class with [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) against the full ~700-player pool.
4. **Train and evaluate.** Both a `DecisionTreeClassifier` and a `LogisticRegression` model (scikit-learn) are trained and scored with a classification report and confusion matrix.
5. **Predict.** The trained model is applied to the full `NBAPlayerStatsRegular.csv` roster to flag which players it classifies as MVP-caliber based on their stat line.
6. **Visualize.** A few exploratory plots (scatter, bar, box plots) chart relationships like age vs. games played and the spread of shooting percentages across the league.

## Data

| File | Rows | Description |
|---|---|---|
| `NBAMVPdataset.csv` | 22 | One row per MVP season, 2001–2022, with the winner's box-score stats for that year |
| `NBAPlayerStatsRegular.csv` | 678 | Full-league regular-season stats for one season, used both as the source of "non-MVP" training examples and as the set of players the trained model predicts on |

## Repo contents

- `MVPPredictor.ipynb` – the full workflow: data loading, labeling, downsampling/SMOTE, model training, evaluation, and visualization.
- `NBAMVPdataset.csv` – historical MVP stats (features + implicit `Yes` label).
- `NBAPlayerStatsRegular.csv` – full-season player stats (source of `No` examples and prediction targets).

## Running the notebook

The notebook was built in Google Colab and uses `google.colab.files.upload()` to load the two CSVs. To run it there:

1. Open `MVPPredictor.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the upload cells and select `NBAMVPdataset.csv` and `NBAPlayerStatsRegular.csv` from this repo when prompted.
3. Run the remaining cells in order.

To run it locally with Jupyter instead, replace the two `files.upload()` cells with `pd.read_csv("NBAMVPdataset.csv")` and `pd.read_csv("NBAPlayerStatsRegular.csv")` (the files are already in this repo, so no upload step is needed), then:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib jupyter
jupyter notebook MVPPredictor.ipynb
```

## Results

- On the small, manually downsampled dataset (22 MVP + 22 non-MVP examples), the decision tree scored ~0.89 accuracy and logistic regression ~0.86 accuracy on the held-out test split.
- After oversampling with SMOTE against the full ~700-player pool, both models scored close to ~0.98–1.00 accuracy on their test split.
- Applied to the full season roster, the final logistic regression model flagged 9 players as MVP-caliber.

## Possible next steps

- Use a strictly held-out season for evaluation (i.e., never train and predict on the same players).
- Add advanced metrics (PER, win shares, VORP, team win %) rather than raw box-score counting stats, since MVP voting weighs team success heavily.
- Cross-validate instead of a single train/test split, given how small the labeled dataset is.
- Compare additional models (random forest, gradient boosting) alongside the decision tree and logistic regression baselines.
