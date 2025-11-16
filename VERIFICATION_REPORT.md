# Pipeline Verification Report

**Generated:** 2025-11-16T12:02:11.304517

**Overall Status:** PASS

## Stage Results

### Stage 1 Exploration

**Status:** PASS

**Checks:**
- raw_data_exists: PASS
- documentation_exists: PASS
- exploration_reports: PASS

### Stage 2 Eda

**Status:** PASS

**Checks:**
- eda_script_exists: PASS
- eda_reports: PASS
- eda_visualizations: PASS

### Stage 3 Dqa

**Status:** PASS

**Checks:**
- dqa_script_exists: PASS
- dqa_reports: PASS
- dqa_visualizations: PASS

### Stage 4 Cleaning

**Status:** PASS

**Checks:**
- cleaning_script_exists: PASS
- cleaned_data_exists: PASS
- cleaning_documentation: PASS

### Stage 5 Feature Engineering

**Status:** PASS

**Checks:**
- fe_script_exists: PASS
- engineered_data_exists: PASS
- fe_reports: FAIL
- fe_visualizations: PARTIAL

**Issues:**
- Feature engineering reports missing

## Data Integrity

**Status:** PASS

**Checks:**
- train_rows_preserved: PASS
- test_rows_preserved: PASS
- column_consistency: FAIL
- data_types: WARN
- missing_values: PASS

**Issues:**
- Column mismatch between train and test

## Recommendations

- Ready for modeling stage
