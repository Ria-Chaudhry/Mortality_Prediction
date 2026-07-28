"""Version-aware non-age-adjusted Charlson/Deyo/Quan coding algorithm.

The code families below implement the Charlson portions of the ICD-9-CM and
ICD-10 algorithm reported by Quan et al., Medical Care 2005,
doi:10.1097/01.mlr.0000182534.19832.83, with the original Charlson weights.
They are expressed independently here as code-family facts; no third-party
software mapping has been copied.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from ..errors import IntegrityError

ALGORITHM_VERSION = "quan-2005-charlson-icd9cm-icd10cm-v1"

WEIGHTS = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "chronic_pulmonary_disease": 1,
    "rheumatic_disease": 1,
    "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1,
    "diabetes_without_complication": 1,
    "diabetes_with_complication": 2,
    "hemiplegia_or_paraplegia": 2,
    "renal_disease": 2,
    "nonmetastatic_malignancy": 2,
    "moderate_or_severe_liver_disease": 3,
    "metastatic_solid_tumor": 6,
    "aids_hiv": 6,
}

# Exact prefixes and inclusive three-character family ranges. Longer, more
# specific prefixes are intentional (for example I252 and 2504).
ICD10_PREFIXES = {
    "myocardial_infarction": ["I21", "I22", "I252"],
    "congestive_heart_failure": [
        "I099",
        "I110",
        "I130",
        "I132",
        "I255",
        "I420",
        "I425",
        "I426",
        "I427",
        "I428",
        "I429",
        "I43",
        "I50",
        "P290",
    ],
    "peripheral_vascular_disease": [
        "I70",
        "I71",
        "I731",
        "I738",
        "I739",
        "I771",
        "I790",
        "I792",
        "K551",
        "K558",
        "K559",
        "Z958",
        "Z959",
    ],
    "cerebrovascular_disease": ["G45", "G46", "H340"],
    "dementia": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    "chronic_pulmonary_disease": [
        "I278",
        "I279",
        "J40",
        "J41",
        "J42",
        "J43",
        "J44",
        "J45",
        "J46",
        "J47",
        "J60",
        "J61",
        "J62",
        "J63",
        "J64",
        "J65",
        "J66",
        "J67",
        "J684",
        "J701",
        "J703",
    ],
    "rheumatic_disease": [
        "M05",
        "M06",
        "M315",
        "M32",
        "M33",
        "M34",
        "M351",
        "M353",
        "M360",
    ],
    "peptic_ulcer_disease": ["K25", "K26", "K27", "K28"],
    "mild_liver_disease": [
        "B18",
        "K700",
        "K701",
        "K702",
        "K703",
        "K709",
        "K713",
        "K714",
        "K715",
        "K717",
        "K73",
        "K74",
        "K760",
        "K762",
        "K763",
        "K764",
        "K768",
        "K769",
        "Z944",
    ],
    "diabetes_without_complication": [
        "E100",
        "E101",
        "E106",
        "E108",
        "E109",
        "E110",
        "E111",
        "E116",
        "E118",
        "E119",
        "E120",
        "E121",
        "E126",
        "E128",
        "E129",
        "E130",
        "E131",
        "E136",
        "E138",
        "E139",
        "E140",
        "E141",
        "E146",
        "E148",
        "E149",
    ],
    "diabetes_with_complication": [
        "E102",
        "E103",
        "E104",
        "E105",
        "E107",
        "E112",
        "E113",
        "E114",
        "E115",
        "E117",
        "E122",
        "E123",
        "E124",
        "E125",
        "E127",
        "E132",
        "E133",
        "E134",
        "E135",
        "E137",
        "E142",
        "E143",
        "E144",
        "E145",
        "E147",
    ],
    "hemiplegia_or_paraplegia": [
        "G041",
        "G114",
        "G801",
        "G802",
        "G81",
        "G82",
        "G830",
        "G831",
        "G832",
        "G833",
        "G834",
        "G839",
    ],
    "renal_disease": [
        "I120",
        "I131",
        "N03",
        "N05",
        "N18",
        "N19",
        "N250",
        "Z490",
        "Z491",
        "Z492",
        "Z940",
        "Z992",
    ],
    "moderate_or_severe_liver_disease": [
        "I850",
        "I859",
        "I864",
        "I982",
        "K704",
        "K711",
        "K721",
        "K729",
        "K765",
        "K766",
        "K767",
    ],
    "aids_hiv": ["B20", "B21", "B22", "B24"],
}

ICD10_RANGES = {
    "cerebrovascular_disease": [("I60", "I69")],
    "nonmetastatic_malignancy": [
        ("C00", "C26"),
        ("C30", "C34"),
        ("C37", "C41"),
        ("C43", "C43"),
        ("C45", "C58"),
        ("C60", "C76"),
        ("C81", "C85"),
        ("C88", "C88"),
        ("C90", "C97"),
    ],
    "metastatic_solid_tumor": [("C77", "C80")],
}

ICD9_PREFIXES = {
    "myocardial_infarction": ["410", "412"],
    "congestive_heart_failure": [
        "39891",
        "40201",
        "40211",
        "40291",
        "40401",
        "40403",
        "40411",
        "40413",
        "40491",
        "40493",
        "4254",
        "4255",
        "4256",
        "4257",
        "4258",
        "4259",
        "428",
    ],
    "peripheral_vascular_disease": [
        "0930",
        "4373",
        "440",
        "441",
        "4431",
        "4432",
        "4438",
        "4439",
        "4471",
        "5571",
        "5579",
        "V434",
    ],
    "cerebrovascular_disease": ["36234"],
    "dementia": ["290", "2941", "3312"],
    "chronic_pulmonary_disease": [
        "4168",
        "4169",
        "490",
        "491",
        "492",
        "493",
        "494",
        "495",
        "496",
        "497",
        "498",
        "499",
        "500",
        "501",
        "502",
        "503",
        "504",
        "505",
        "5064",
        "5081",
        "5088",
    ],
    "rheumatic_disease": [
        "4465",
        "7100",
        "7101",
        "7102",
        "7103",
        "7104",
        "7140",
        "7141",
        "7142",
        "7148",
        "725",
    ],
    "peptic_ulcer_disease": ["531", "532", "533", "534"],
    "mild_liver_disease": [
        "07022",
        "07023",
        "07032",
        "07033",
        "07044",
        "07054",
        "0706",
        "0709",
        "570",
        "571",
        "5733",
        "5734",
        "5738",
        "5739",
        "V427",
    ],
    "diabetes_without_complication": [
        "2500",
        "2501",
        "2502",
        "2503",
        "2508",
        "2509",
    ],
    "diabetes_with_complication": ["2504", "2505", "2506", "2507"],
    "hemiplegia_or_paraplegia": [
        "3341",
        "342",
        "343",
        "3440",
        "3441",
        "3442",
        "3443",
        "3444",
        "3445",
        "3446",
        "3449",
    ],
    "renal_disease": [
        "40301",
        "40311",
        "40391",
        "40402",
        "40403",
        "40412",
        "40413",
        "40492",
        "40493",
        "582",
        "5830",
        "5831",
        "5832",
        "5833",
        "5834",
        "5835",
        "5836",
        "5837",
        "585",
        "586",
        "5880",
        "V420",
        "V451",
        "V56",
    ],
    "nonmetastatic_malignancy": ["2386"],
    "moderate_or_severe_liver_disease": [
        "4560",
        "4561",
        "4562",
        "5722",
        "5723",
        "5724",
        "5725",
        "5726",
        "5727",
        "5728",
    ],
}

ICD9_RANGES = {
    "cerebrovascular_disease": [("430", "438")],
    "nonmetastatic_malignancy": [
        ("140", "172"),
        ("174", "195"),
        ("200", "208"),
    ],
    "metastatic_solid_tumor": [("196", "199")],
    "aids_hiv": [("042", "044")],
}


@dataclass(frozen=True)
class CharlsonResult:
    categories: frozenset[str]
    score: int


def normalize_icd_code(code: object) -> str:
    """Normalize case/spacing/punctuation while preserving every alphanumeric."""
    return "".join(character for character in str(code).upper().strip() if character.isalnum())


def classify_icd(code: object, icd_version: object) -> set[str]:
    normalized = normalize_icd_code(code)
    try:
        version = int(icd_version)
    except (TypeError, ValueError):
        raise IntegrityError(f"Invalid or missing ICD version for code {code!r}") from None
    if version not in {9, 10}:
        raise IntegrityError(f"Unsupported ICD version {version!r} for code {code!r}")
    prefixes = ICD9_PREFIXES if version == 9 else ICD10_PREFIXES
    ranges = ICD9_RANGES if version == 9 else ICD10_RANGES
    categories = {
        category
        for category, values in prefixes.items()
        if any(normalized.startswith(prefix) for prefix in values)
    }
    family = normalized[:3]
    categories.update(
        category
        for category, intervals in ranges.items()
        if any(start <= family <= end for start, end in intervals)
    )
    return categories


def score_categories(categories: Iterable[str]) -> CharlsonResult:
    retained = set(categories)
    hierarchy = [
        ("diabetes_with_complication", "diabetes_without_complication"),
        ("moderate_or_severe_liver_disease", "mild_liver_disease"),
        ("metastatic_solid_tumor", "nonmetastatic_malignancy"),
    ]
    for severe, mild in hierarchy:
        if severe in retained:
            retained.discard(mild)
    unknown = retained - set(WEIGHTS)
    if unknown:
        raise IntegrityError(f"Unknown Charlson categories: {sorted(unknown)}")
    return CharlsonResult(
        categories=frozenset(retained),
        score=sum(WEIGHTS[category] for category in retained),
    )


def score_diagnosis_frame(frame: pd.DataFrame) -> CharlsonResult:
    required = {"code", "icd_version"}
    missing = required - set(frame)
    if missing:
        raise IntegrityError(f"Charlson diagnoses missing columns: {sorted(missing)}")
    categories: set[str] = set()
    for row in frame[["code", "icd_version"]].drop_duplicates().itertuples(index=False):
        categories.update(classify_icd(row.code, row.icd_version))
    return score_categories(categories)
