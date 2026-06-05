import numpy as np
import pandas as pd

from mosek.fusion import Domain, Expr, Model, Var
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


def resolve_feature_names(X, features, *, default):
    """Resolve a column selector against a pandas DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("resolve_feature_names requires X to be a pandas DataFrame.")

    if features is None:
        return list(default)

    if callable(features):
        features = features(X)

    if isinstance(features, (str, bytes)):
        features = [features]

    resolved = []
    columns = X.columns

    for feature in features:
        if isinstance(feature, (int, np.integer)):
            try:
                resolved.append(columns[int(feature)])
            except IndexError as exc:
                raise ValueError(
                    f"Column index {feature} is out of bounds for "
                    f"{len(columns)} columns."
                ) from exc
        else:
            if feature not in columns:
                raise ValueError(f"Unknown feature name: {feature!r}.")
            resolved.append(feature)

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Feature selector contains duplicates: {resolved!r}.")

    return resolved

def make_preprocessor(
    X,
    numerical_features=None,
    categorical_features=None
    ) -> ColumnTransformer:
    """Construct an order-preserving preprocessing pipeline.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("make_preprocessor requires X to be a pandas DataFrame.")

    columns = list(X.columns)

    categorical_names = resolve_feature_names(
        X,
        categorical_features,
        default=[]
    )
    categorical_set = set(categorical_names)

    if numerical_features is None:
        numerical_default = [col for col in columns if col not in categorical_set]
    else:
        numerical_default = []

    numerical_names = resolve_feature_names(
        X,
        numerical_features,
        default=numerical_default,
    )
    numerical_set = set(numerical_names)

    overlap = numerical_set & categorical_set
    if overlap:
        raise ValueError(
            "numerical_features and categorical_features must not overlap. "
            f"Overlap: {sorted(overlap)!r}."
        )

    covered = numerical_set | categorical_set
    missing = [col for col in columns if col not in covered]
    if missing:
        raise ValueError(
            "Every column must be assigned to exactly one preprocessing type. "
            f"Unassigned columns: {missing!r}."
        )

    transformers = []
    for col in columns:
        if col in numerical_set:
            pipe = Pipeline([
                # IterativeImputer: higher variance, still experimental
                ("numerical_imputer", SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                )),
                ("numerical_scaler", RobustScaler()),
            ])
        elif col in categorical_set:
            pipe = Pipeline([
                # OneHotEncoder: higher dimensionality, requires group selection
                ("categorical_encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                    min_frequency=2,
                )),
            ])

        transformers.append((f"col_{len(transformers)}", pipe, [col]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

def resolve_subsample_size(
        value: int | float,
        maximum: int
    ) -> int:
    """Resolve an absolute or relative subsample size. If ``value`` is a float in
    ``(0, 1]``, it is interpreted as a fraction of ``maximum``. Otherwise, it is
    interpreted as an absolute subsample size.
    """
    if isinstance(value, float) and 0 < value <= 1:
        size = int(np.ceil(value * maximum))
    else:
        size = int(value)

    if not 1 <= size <= maximum:
        raise ValueError(
            f"Subsample size must be in [1, {maximum}]. Got {size}."
        )

    return size

def softplus(
        M: "Model",
        t: "Var",
        u: "Expr"
    ) -> None:
    """
    This is the MOSEK Fusion exponential cone representation of the softplus function
    (https://docs.mosek.com/latest/pythonfusion/case-studies-logistic.html).
    """
    n = t.getShape()[0]

    z1 = M.variable(n)
    z2 = M.variable(n)

    M.constraint(Expr.add(z1, z2), Domain.equalsTo(1.0))

    M.constraint(
        Expr.hstack(z1, Expr.constTerm(n, 1.0), Expr.sub(u, t)), Domain.inPExpCone(),
    )

    M.constraint(
        Expr.hstack(z2, Expr.constTerm(n, 1.0), Expr.neg(t)), Domain.inPExpCone(),
    )
