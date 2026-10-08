from __future__ import annotations

from math import sqrt

from haversine import haversine

from entilink import predicates
from entilink.variables.base import FieldType


class LatLongType(FieldType):
    type = "LatLong"

    _predicate_functions = [predicates.latLongGridPredicate]

    @staticmethod
    def comparator(x: tuple[float, float], y: tuple[float, float]) -> float:
        return sqrt(haversine(x, y))
