"""Constante comune pentru pipeline-ul U19."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

YEARS = (2017, 2025)
CATEGORIES = ("flood", "fire", "storm")

# Europa geografică; Türkiye, Georgia, Armenia și teritoriile de peste mări (Mayotte etc.) sunt excluse.
EUROPE = {
    "Albania", "Andorra", "Austria", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia",
    "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Iceland", "Ireland", "Italy", "Kosovo", "Kosovo*", "Latvia", "Liechtenstein", "Lithuania",
    "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Norway", "Poland", "Portugal", "Romania", "San Marino", "Serbia", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom",
}

EUROPE_ISO3 = {
    "ALB", "AND", "AUT", "BEL", "BIH", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU",
    "GRC", "HUN", "ISL", "IRL", "ITA", "XKO", "LVA", "LIE", "LTU", "LUX", "MLT", "MDA", "MCO", "MNE",
    "NLD", "MKD", "NOR", "POL", "PRT", "ROU", "SMR", "SRB", "SVK", "SVN", "ESP", "SWE", "CHE", "UKR", "GBR",
}

GEE_PROJECT = "thinking-glass-425111-v0"

# ferestrele față de data de referință a evenimentului (zile); decise la CP0
PRE_WINDOW = (-60, -10)
POST_WINDOW = (2, 30)
