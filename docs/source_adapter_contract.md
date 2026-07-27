# Source adapter contract

Both adapters return `StandardizedData(tables, input_hashes, mapping_hash, audit)`. Table keys and
required fields are:

| Table | Fields |
|---|---|
| `patients` | `patient_id`, `birth_datetime`, `sex`, `race`, `ethnicity` |
| `encounters` | `visit_id`, `patient_id`, `start_datetime`, `end_datetime`, `visit_type`, `elective`, `followup_end_datetime` |
| `deaths` | `patient_id`, `death_datetime` |
| `diagnoses` | `diagnosis_id`, `visit_id`, `patient_id`, `diagnosis_datetime`, `code`, `source_table` |
| domain event | `event_id`, `source_visit_id`, `bridge_key`, `patient_id`, `event_datetime`, `concept_key`, `concept_name`, `value`, `unit`, `source_table`, `semantics` |
| `bridge` | `bridge_key`, `visit_id` |
| `metadata` | `domain`, `concept_key`, `concept_name`, `source_table`, `semantics`, `unit` |

Patient, visit, and event keys must be unique within their table. Dates are parsed before common
logic. Missing optional direct/bridge keys are allowed; missing clinical event times or semantics
are not.

Linkage precedence is confirmed direct visit, approved bridge, then patient plus time. Patient
identity must agree for direct/bridge matches. Patient-time linkage requires exactly one candidate
whose interval contains the event. Multiple candidates hard-fail; unmatched rows are counted and
excluded.

The CHoRUS adapter reads configured OMOP-compatible person, visit, death, condition, measurement,
drug, procedure, observation, and bridge structures from local files or read-only SQL. SQL
credentials come only from a named environment variable. Observation structures are retained as
their real semantics and can be mapped as measurement candidates only after a confirmed numeric
mapping; categorical observations are not coerced.

The default MIMIC adapter maps admissions, patients, recorded deaths, diagnoses ICD, labevents,
prescriptions, and procedures ICD. Prescriptions are prescriptions/orders, not administrations;
procedures ICD are coded procedures, not necessarily bedside performed events. Users may configure
alternative eMAR/inputevents/procedureevents sources only with explicit semantics. CSV, CSV.GZ,
and Parquet are accepted.

Discovery/validation records source table, column, semantics, mapping hash, input checksum, and
row count without publishing patient rows.
