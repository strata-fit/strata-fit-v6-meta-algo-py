import warnings

# Suppress known upstream deprecation warnings emitted by dependencies
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"dateutil\.tz",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"sqlalchemy\.sql\.sqltypes",
)
