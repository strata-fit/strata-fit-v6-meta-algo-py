from __future__ import annotations

from datetime import date
from typing import Optional, Union

import pandas as pd

from config.config import settings
from pydantic import BaseModel, Field, ValidationError, create_model


class ValidationDetail(BaseModel):
    row: Union[int, str]
    field: str
    message: str
    error_type: str
    input_value: str


def load_data_models_from_settings() -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for model_name, fields in settings.schema.pydantic.items():
        model_fields = {}
        for field_name, field in fields.items():
            field_type = eval(field["type"])
            constraints = {}
            if "ge" in field:
                constraints["ge"] = field["ge"]
            if "le" in field:
                constraints["le"] = field["le"]
            if "min_length" in field:
                constraints["min_length"] = field["min_length"]
            if "max_length" in field:
                constraints["max_length"] = field["max_length"]
            if "pattern" in field:
                constraints["pattern"] = field["pattern"]
            if "regex" in field:
                constraints["pattern"] = field["regex"]

            model_fields[field_name] = (field_type, Field(..., **constraints))

        models[model_name] = create_model(model_name, **model_fields)
    return models


def get_date_fields(model: type[BaseModel]) -> list[str]:
    return [
        field_name
        for field_name, field_type in model.__annotations__.items()
        if field_type == date
    ]


def validate_csv(
    df: pd.DataFrame,
    model: type[BaseModel],
) -> tuple[bool, list[ValidationDetail]]:
    date_fields = get_date_fields(model)
    for field in date_fields:
        if field in df.columns:
            df[field] = pd.to_datetime(df[field], errors="coerce")
            df[field] = df[field].apply(lambda value: value if pd.notnull(value) else None)

    errors: list[ValidationDetail] = []
    for index, row in df.iterrows():
        try:
            row_dict = row.where(pd.notnull(row), None).to_dict()
            model(**row_dict)
        except ValidationError as exc:
            errors.extend(translate_errors(exc.errors(), index))
    return len(errors) > 0, errors


def _lookup_error_message(field: str, error_type: str, fallback: str) -> str:
    try:
        error_messages = settings.schema.error_messages
    except AttributeError:
        error_messages = {}

    for error_message in error_messages.get(field, []):
        if error_message.get("type") == error_type:
            return error_message.get("message", fallback)
    return fallback


def translate_errors(errors, row: int | str) -> list[ValidationDetail]:
    readable_errors: list[ValidationDetail] = []
    for error in errors:
        field = str(error["loc"][0])
        error_type = error["type"]
        input_value = str(error.get("input", "N/A"))
        message = _lookup_error_message(field, error_type, error["msg"])
        readable_errors.append(
            ValidationDetail(
                row=row,
                field=field,
                message=message,
                error_type=error_type,
                input_value=input_value,
            )
        )
    return readable_errors
