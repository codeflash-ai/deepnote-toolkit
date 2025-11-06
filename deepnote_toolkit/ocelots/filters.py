from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class FilterOperator(str, Enum):
    IS_EQUAL = "is-equal"
    IS_NOT_EQUAL = "is-not-equal"
    TEXT_CONTAINS = "text-contains"
    TEXT_DOES_NOT_CONTAIN = "text-does-not-contain"
    GREATER_THAN = "greater-than"
    GREATER_THAN_OR_EQUAL = "greater-than-or-equal"
    LESS_THAN = "less-than"
    LESS_THAN_OR_EQUAL = "less-than-or-equal"
    OUTSIDE_OF = "outside-of"
    IS_ONE_OF = "is-one-of"
    IS_NOT_ONE_OF = "is-not-one-of"
    IS_NULL = "is-null"
    IS_NOT_NULL = "is-not-null"
    BETWEEN = "between"
    IS_AFTER = "is-after"
    IS_BEFORE = "is-before"
    IS_ON = "is-on"
    IS_RELATIVE_TODAY = "is-relative-today"


@dataclass(frozen=True)
class Filter:
    column: str
    operator: FilterOperator
    comparative_values: List[Any]

    # We need class to have __eq__ and __hash__ to be able to compare instances and use
    # filters in dict (as keys) and sets
    def __eq__(self, other):
        if not isinstance(other, Filter):
            return False
        return (
            self.column == other.column
            and self.operator == other.operator
            and self.comparative_values == other.comparative_values
        )

    def __hash__(self):
        return hash((self.column, self.operator, tuple(self.comparative_values)))

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> Filter:
        """Create a Filter instance from a dictionary.

        Args:
            input_dict: Dictionary containing filter specification with keys:
                       - column: Column name to filter on
                       - operator: Filter operator (must match FilterOperator enum)
                       - comparativeValues: List of values to compare against
                       OR (for legacy contains filter)
                       - id: Column name to filter on
                       - value: Value to search for
                       - type: "contains" for text contains filter

        Returns:
            Filter: New Filter instance

        Raises:
            ValueError: If required keys are missing or operator is invalid
        """
        # Handle legacy column contains filter format
        if input_dict.get("type") == "contains":
            try:
                column = input_dict["id"]
                value = input_dict["value"]
            except KeyError:
                raise ValueError(
                    "Missing required keys for contains filter: id and value"
                )
            return cls(
                column=column,
                operator=FilterOperator.TEXT_CONTAINS,
                comparative_values=[value],
            )

        try:
            column = input_dict["column"]
            operator_raw = input_dict["operator"]
            comparative_values = input_dict["comparativeValues"]
        except KeyError:
            # Avoid building a list (original allocates unnecessary list)
            raise ValueError(
                "Missing required keys: ['column', 'operator', 'comparativeValues']"
            )

        try:
            operator = FilterOperator(operator_raw)
        except ValueError:
            raise ValueError(f"Invalid operator: {operator_raw}")

        return cls(
            column=column,
            operator=operator,
            comparative_values=comparative_values,
        )
