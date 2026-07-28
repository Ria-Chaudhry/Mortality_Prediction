# Charlson implementation

The pipeline implements the non-age-adjusted Charlson categories and original weights using
separate ICD-9-CM and ICD-10-CM rules derived from:

- Quan H, et al. Coding Algorithms for Defining Comorbidities in ICD-9-CM and ICD-10
  Administrative Data. *Medical Care*. 2005;43(11):1130-1139.
  doi:10.1097/01.mlr.0000182534.19832.83.
- Deyo RA, Cherkin DC, Ciol MA. Adapting a clinical comorbidity index for use with ICD-9-CM
  administrative databases. *J Clin Epidemiol*. 1992;45(6):613-619.
  doi:10.1016/0895-4356(92)90133-8.

Algorithm version is `quan-2005-charlson-icd9cm-icd10cm-v1`. Codes are uppercased and punctuation
removed while all alphanumeric specificity is retained. `icd_version` must be 9 or 10.

The complete implemented categories are myocardial infarction, congestive heart failure,
peripheral vascular disease, cerebrovascular disease, dementia, chronic pulmonary disease,
rheumatic disease, peptic ulcer disease, mild liver disease, uncomplicated diabetes,
complicated diabetes, hemiplegia/paraplegia, renal disease, nonmetastatic malignancy,
moderate/severe liver disease, metastatic solid tumor, and AIDS/HIV.

Hierarchies remove uncomplicated diabetes when complicated diabetes is present, mild liver
disease when moderate/severe disease is present, and nonmetastatic malignancy when metastatic
disease is present. Duplicate diagnoses and repeated categories do not add weight twice.

Only diagnoses attached to prior qualifying acute admissions in the configured lookback are
scored. The index admission is excluded even when it contains a code. Boundary, hierarchy,
duplicate, multi-admission, ICD-9, and ICD-10 behavior is tested.

The rule implementation was independently expressed as coding-family facts in
`cohort/charlson.py`; no third-party software mapping was copied. Category/version/provenance
metadata is in `mappings/charlson_mapping.csv`.
