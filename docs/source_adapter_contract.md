# Source adapter contract

Both adapters return `StandardizedData(tables, input_hashes, mapping_hash, audit)`. Common
analytical modules accept only these standardized frames.

| Table | Required standardized fields |
|---|---|
| `patients` | `patient_id`, `birth_datetime`, `sex`, `race`, `ethnicity`; optional/common audit fields `age_anchor`, `age_anchor_year`, `anchor_year_group` |
| `encounters` | `visit_id`, `patient_id`, `start_datetime`, `end_datetime`, `visit_type`, `elective`, `followup_end_datetime`, `race_at_admission`, `ethnicity_at_admission` |
| `deaths` | `patient_id`, exact `death_datetime` when available, date-only `death_date`, `death_time_precision`, `death_source`, `death_source_conflict` |
| `diagnoses` | `diagnosis_id`, `visit_id`, `patient_id`, `diagnosis_datetime`, `code`, `icd_version`, `source_table` |
| domain event | `event_id`, `source_visit_id`, `bridge_key`, `patient_id`, exact `event_datetime` or date-only `event_date`, `event_time_precision`, `concept_key`, `concept_name`, `value`, `unit`, `source_table`, `semantics` |
| `bridge` | `bridge_key`, `visit_id` |
| `metadata` | `domain`, `concept_key`, `concept_name`, `source_table`, `semantics`, `unit` |

Patient and visit keys are unique. Event keys are unique without deduplicating multiplicity.
Linkage precedence is confirmed direct visit, approved bridge, then patient plus time. Direct and
bridge links must agree with patient identity. Patient-time linkage requires exactly one
eligible interval; ambiguity hard-fails. Unmatched rows are audited and excluded.

## CHoRUS

The configurable OMOP-compatible adapter supports person, visit, death, condition, measurement,
drug exposure, procedure occurrence, observation, and approved bridge structures. Physical
tables and every source column come from configuration. SQL uses explicit projections, a
server-side temporary acute-cohort relation populated in bounded batches, and correlated
cohort/time predicates. It does not expand one SQL parameter per cohort member. Database URL and
schema values come from environment variables.

Medication and procedure semantics are preserved per row or configured source. Drug exposure is
not silently called administration; a claim, code, or order is not silently called performed
care. Observations remain audit-only unless a confirmed numeric mapping explicitly admits them.

## Native MIMIC-IV

The tested native path requires:

| Source | Native columns used |
|---|---|
| `patients` | `subject_id`, `gender`, `anchor_age`, `anchor_year`, `anchor_year_group`, `dod` |
| `admissions` | `subject_id`, `hadm_id`, `admittime`, `dischtime`, `deathtime`, `admission_type`, `race` |
| `diagnoses_icd` | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version` |
| `labevents` | `labevent_id`, `subject_id`, `hadm_id`, `itemid`, `charttime`, `valuenum`, `valueuom` |
| optional `chartevents` | `subject_id`, `hadm_id`, `itemid`, `charttime`, `valuenum`, `valueuom`; stable internal event key |
| `prescriptions` | `subject_id`, `hadm_id`, `pharmacy_id`, `poe_id`, `poe_seq`, `starttime`, `stoptime`, `drug`, `formulary_drug_cd`, `gsn`, `ndc` |
| `procedures_icd` | `subject_id`, `hadm_id`, `seq_num`, `chartdate`, `icd_code`, `icd_version` |

All required columns are validated before analysis. The adapter does not require
`anchor_birth_datetime`, `followup_end_datetime`, `admission_type_normalized`,
`medication_event_id`, or `procedure_event_id`.

Age follows the MIMIC anchor method. Race, ethnicity availability/derivation, and admission types
require configured harmonization.
The death/follow-up rule is explicit. Precise `admissions.deathtime` takes priority;
`patients.dod` is retained as a date-only fallback and never converted to midnight. Source
conflicts are audited, and a date-only landmark-day death is conservatively excluded without
inventing a time. Medication concept field and source semantics are explicit.
Measurement concepts are namespaced as `labevents:itemid` or `chartevents:itemid`; procedures use
the recovered `icd{version}:{normalized code}` namespace. `procedures_icd.chartdate` stays date-only and uses
the recovered `calendar_dates_spanned_inclusive_v1` rule: admission through predictor-end
calendar dates, inclusive. It is not described as an exact timestamp window. Stable internal keys hash native key material and add
a duplicate occurrence index, preserving duplicate-row multiplicity.

Parquet uses column and predicate pushdown. CSV/CSV.GZ is read with `usecols` in bounded chunks,
then candidate `hadm_id`/`subject_id` and global time predicates are applied before pandas
concatenation. Domain frames are never loaded with unrestricted whole-file `pandas.read_*`.

The exact manuscript MIMIC release and follow-up choice remain unconfirmed. Native compatibility
does not imply that an arbitrary MIMIC release reproduces the paper cohort.
