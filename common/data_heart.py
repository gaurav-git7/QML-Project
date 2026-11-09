"""Heart-disease (CDC BRFSS 2020 survey) data: 16 numeric features, scaling to [0,1], top-12 feature selection.

The same functions turn CSV rows and dashboard form entries into model inputs, so training and inference agree."""
import numpy as np
import pandas as pd

import config as C

AGE_ORDER = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
             "60-64", "65-69", "70-74", "75-79", "80 or older"]
GEN_ORDER = ["Poor", "Fair", "Good", "Very good", "Excellent"]
DIABETIC_ORDER = ["No", "No, borderline diabetes", "Yes (during pregnancy)", "Yes"]
_DIAB_CODE = {"No": 0, "No, borderline diabetes": 1, "Yes (during pregnancy)": 1, "Yes": 2}

# (column, type, label) - the 16 features; type: num | yn | sex | age | gen | diab
FEATURES = [
    ("BMI", "num", "Body-mass index"), ("Smoking", "yn", "Smoked >= 100 cigarettes in life"),
    ("AlcoholDrinking", "yn", "Heavy alcohol drinking"), ("Stroke", "yn", "Ever had a stroke"),
    ("PhysicalHealth", "num", "Days physical health not good (last 30)"),
    ("MentalHealth", "num", "Days mental health not good (last 30)"),
    ("DiffWalking", "yn", "Difficulty walking / climbing stairs"), ("Sex", "sex", "Sex"),
    ("AgeCategory", "age", "Age category"), ("Diabetic", "diab", "Diabetes"),
    ("PhysicalActivity", "yn", "Physical activity in last 30 days"), ("GenHealth", "gen", "General health"),
    ("SleepTime", "num", "Sleep (hours per night)"), ("Asthma", "yn", "Asthma"),
    ("KidneyDisease", "yn", "Kidney disease"), ("SkinCancer", "yn", "Skin cancer"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]


def encode_frame(df):
    """Raw survey columns (strings / numbers as in the CSV) -> (N,16) float32."""
    cols = []
    for name, kind, _ in FEATURES:
        s = df[name]
        if kind == "num":
            v = s.astype(float)
        elif kind == "yn":
            v = (s == "Yes").astype(float)
        elif kind == "sex":
            v = (s == "Male").astype(float)
        elif kind == "age":
            v = s.map({a: i for i, a in enumerate(AGE_ORDER)}).astype(float)
        elif kind == "gen":
            v = s.map({g: i for i, g in enumerate(GEN_ORDER)}).astype(float)
        else:
            v = s.map(_DIAB_CODE).astype(float)
        cols.append(v.values)
    return np.stack(cols, axis=1).astype(np.float32)


def load_heart():
    """-> (X16 raw float32, y int array)."""
    raw = pd.read_csv(C.HEART_CSV)
    return encode_frame(raw), (raw["HeartDisease"] == "Yes").astype(int).values


class HeartProcessor:
    """Scales the 16 raw features to [0,1] (1st-99th percentile of the training data, clipped) and keeps the
    indices of the 12 most informative features (mutual information on the training subset)."""

    def fit(self, X_raw_train):
        self.lo = np.percentile(X_raw_train, 1, axis=0)
        self.hi = np.percentile(X_raw_train, 99, axis=0)
        self.top12 = np.arange(12)
        return self

    def scale(self, X_raw):
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return np.clip((X_raw - self.lo) / span, 0, 1).astype(np.float32)

    def rep(self, X_scaled, name):
        return X_scaled if name == "H16" else X_scaled[:, self.top12]


def row_to_raw(values):
    """dict of dashboard form values (strings / numbers keyed by column name) -> (1,16) raw features."""
    return encode_frame(pd.DataFrame([values]))
