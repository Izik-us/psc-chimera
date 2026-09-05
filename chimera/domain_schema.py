"""Explicit NRPS/PKS domain schemas used by conditioning and validation."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class DomainType(str, Enum):
    A = "A"
    T = "T"
    C = "C"
    E = "E"
    CY = "Cy"
    TE = "TE"
    MT = "Mt"
    KS = "KS"
    AT = "AT"
    ACP = "ACP"
    KR = "KR"
    DH = "DH"
    ER = "ER"
    LINKER = "linker"
    INSERT = "insert"


@dataclass(frozen=True)
class DomainSpan:
    start: int
    end: int
    domain_type: DomainType

    def validate(self, length: int) -> None:
        if not 0 <= self.start < self.end <= length:
            raise ValueError(f"Invalid domain span {self.start}:{self.end} for length {length}")


@dataclass(frozen=True)
class AssemblySchema:
    domains: Sequence[DomainSpan]
    module_boundaries: Sequence[tuple[int, int]]

    def validate(self, length: int) -> None:
        for domain in self.domains:
            domain.validate(length)
        for start, end in self.module_boundaries:
            if not 0 <= start < end <= length:
                raise ValueError(f"Invalid module span {start}:{end}")
