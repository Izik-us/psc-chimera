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
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("domain boundaries must be integers")
        if not 0 <= self.start < self.end <= length:
            raise ValueError(f"Invalid domain span {self.start}:{self.end} for length {length}")


@dataclass(frozen=True)
class AssemblySchema:
    domains: Sequence[DomainSpan]
    module_boundaries: Sequence[tuple[int, int]]

    def validate(self, length: int) -> None:
        if length < 1:
            raise ValueError("assembly length must be positive")

        domains = sorted(self.domains, key=lambda d: (d.start, d.end))
        for domain in domains:
            domain.validate(length)
        for previous, current in zip(domains, domains[1:]):
            if current.start < previous.end:
                raise ValueError(
                    f"Overlapping domains {previous.start}:{previous.end} and "
                    f"{current.start}:{current.end}"
                )

        modules = sorted(self.module_boundaries)
        for start, end in modules:
            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError("module boundaries must be integer pairs")
            if not 0 <= start < end <= length:
                raise ValueError(f"Invalid module span {start}:{end}")
        for previous, current in zip(modules, modules[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"Overlapping modules {previous[0]}:{previous[1]} and "
                    f"{current[0]}:{current[1]}"
                )

        # Every declared domain must belong to exactly one module. This keeps
        # scale-2 domain pooling and scale-3 module pooling unambiguous.
        for domain in domains:
            containing = sum(
                module_start <= domain.start and domain.end <= module_end
                for module_start, module_end in modules
            )
            if containing != 1:
                raise ValueError(
                    f"Domain {domain.start}:{domain.end} must be contained in exactly one module"
                )
