"""
Comprehensive Verification Agent for Data Analysis Pipeline
============================================================

This agent verifies all completed work in the data analysis pipeline:
1. Problem Understanding & Data Exploration
2. Exploratory Data Analysis (EDA)
3. Data Quality Assessment (DQA)
4. Data Cleaning
5. Feature Engineering

Author: Verification Agent
Date: 2025-11-16
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class VerificationAgent:
    """Comprehensive verification agent for data pipeline validation"""
    
    def __init__(self, base_dir=None):
        """Initialize verification agent"""
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.data_dir = self.base_dir / 'data'
        self.reports_dir = self.base_dir / 'reports'
        self.scripts_dir = self.base_dir / 'scripts'
        self.docs_dir = self.base_dir / 'documentation'
        self.viz_dir = self.base_dir / 'visualizations'
        
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'overall_status': 'PENDING',
            'issues': [],
            'recommendations': []
        }
    
    def verify_stage_1_exploration(self):
        """Verify Stage 1: Problem Understanding & Data Exploration"""
        print("\n" + "="*80)
        print("STAGE 1: Problem Understanding & Data Exploration")
        print("="*80)
        
        stage_results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check 1.1: Raw data files exist
        print("\n[1.1] Verifying raw data files...")
        train_file = self.data_dir / 'data_minihackathon_train.csv'
        test_file = self.data_dir / 'data_minihackathon_test.csv'
        
        if train_file.exists() and test_file.exists():
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            stage_results['checks']['raw_data_exists'] = 'PASS'
            print(f"   ✓ Train data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
            print(f"   ✓ Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
        else:
            stage_results['checks']['raw_data_exists'] = 'FAIL'
            stage_results['issues'].append("Raw data files missing")
            stage_results['status'] = 'FAIL'
            print("   ✗ Raw data files not found")
        
        # Check 1.2: Documentation exists
        print("\n[1.2] Verifying documentation...")
        doc_files = ['data_overview.md', 'attributes.md']
        doc_status = []
        
        for doc in doc_files:
            doc_path = self.docs_dir / doc
            if doc_path.exists():
                doc_status.append(f"✓ {doc}")
            else:
                doc_status.append(f"✗ {doc} (missing)")
                stage_results['issues'].append(f"Missing documentation: {doc}")
        
        for status in doc_status:
            print(f"   {status}")
        
        stage_results['checks']['documentation_exists'] = 'PASS' if all('✓' in s for s in doc_status) else 'PARTIAL'
        
        # Check 1.3: Exploration reports
        print("\n[1.3] Verifying exploration reports...")
        exploration_dir = self.reports_dir / 'exploration'
        
        if exploration_dir.exists():
            exploration_files = list(exploration_dir.glob('*.md')) + list(exploration_dir.glob('*.txt'))
            if exploration_files:
                stage_results['checks']['exploration_reports'] = 'PASS'
                print(f"   ✓ Found {len(exploration_files)} exploration report(s)")
            else:
                stage_results['checks']['exploration_reports'] = 'FAIL'
                stage_results['issues'].append("No exploration reports found")
                print("   ✗ No exploration reports found")
        else:
            stage_results['checks']['exploration_reports'] = 'FAIL'
            stage_results['issues'].append("Exploration directory missing")
            print("   ✗ Exploration directory not found")
        
        self.verification_results['stages']['stage_1_exploration'] = stage_results
        print(f"\nStage 1 Status: {stage_results['status']}")
        return stage_results
    
    def verify_stage_2_eda(self):
        """Verify Stage 2: Exploratory Data Analysis"""
        print("\n" + "="*80)
        print("STAGE 2: Exploratory Data Analysis (EDA)")
        print("="*80)
        
        stage_results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check 2.1: EDA script exists
        print("\n[2.1] Verifying EDA script...")
        eda_script = self.scripts_dir / 'eda_analysis.py'
        
        if eda_script.exists():
            stage_results['checks']['eda_script_exists'] = 'PASS'
            print(f"   ✓ EDA script found: {eda_script.name}")
            
            # Check script completeness
            with open(eda_script, 'r', encoding='utf-8') as f:
                content = f.read()
                required_analyses = ['univariate', 'bivariate', 'multivariate', 'statistical']
                found_analyses = [a for a in required_analyses if a in content.lower()]
                print(f"   ✓ Found {len(found_analyses)}/{len(required_analyses)} analysis types")
        else:
            stage_results['checks']['eda_script_exists'] = 'FAIL'
            stage_results['issues'].append("EDA script missing")
            stage_results['status'] = 'FAIL'
            print("   ✗ EDA script not found")
        
        # Check 2.2: EDA reports
        print("\n[2.2] Verifying EDA reports...")
        eda_reports_dir = self.reports_dir / 'eda'
        
        if eda_reports_dir.exists():
            report_files = list(eda_reports_dir.glob('*.md')) + list(eda_reports_dir.glob('*.txt'))
            if report_files:
                stage_results['checks']['eda_reports'] = 'PASS'
                print(f"   ✓ Found {len(report_files)} EDA report(s)")
                for rf in report_files:
                    print(f"      - {rf.name}")
            else:
                stage_results['checks']['eda_reports'] = 'FAIL'
                stage_results['issues'].append("No EDA reports found")
                print("   ✗ No EDA reports found")
        else:
            stage_results['checks']['eda_reports'] = 'FAIL'
            stage_results['issues'].append("EDA reports directory missing")
            print("   ✗ EDA reports directory not found")
        
        # Check 2.3: EDA visualizations
        print("\n[2.3] Verifying EDA visualizations...")
        eda_viz_dir = self.viz_dir / 'eda'
        
        if eda_viz_dir.exists():
            viz_files = list(eda_viz_dir.glob('*.png')) + list(eda_viz_dir.glob('*.jpg'))
            if viz_files:
                stage_results['checks']['eda_visualizations'] = 'PASS'
                print(f"   ✓ Found {len(viz_files)} visualization(s)")
            else:
                stage_results['checks']['eda_visualizations'] = 'PARTIAL'
                stage_results['issues'].append("No visualizations found (may be expected)")
                print("   ⚠ No visualizations found")
        else:
            stage_results['checks']['eda_visualizations'] = 'PARTIAL'
            print("   ⚠ EDA visualizations directory not found")
        
        self.verification_results['stages']['stage_2_eda'] = stage_results
        print(f"\nStage 2 Status: {stage_results['status']}")
        return stage_results
    
    def verify_stage_3_dqa(self):
        """Verify Stage 3: Data Quality Assessment"""
        print("\n" + "="*80)
        print("STAGE 3: Data Quality Assessment (DQA)")
        print("="*80)
        
        stage_results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check 3.1: DQA script exists
        print("\n[3.1] Verifying DQA script...")
        dqa_script = self.scripts_dir / 'data_quality_assessment.py'
        
        if dqa_script.exists():
            stage_results['checks']['dqa_script_exists'] = 'PASS'
            print(f"   ✓ DQA script found: {dqa_script.name}")
            
            # Check script content
            with open(dqa_script, 'r', encoding='utf-8') as f:
                content = f.read()
                required_checks = ['missing', 'outlier', 'duplicate', 'consistency']
                found_checks = [c for c in required_checks if c in content.lower()]
                print(f"   ✓ Found {len(found_checks)}/{len(required_checks)} quality checks")
        else:
            stage_results['checks']['dqa_script_exists'] = 'FAIL'
            stage_results['issues'].append("DQA script missing")
            stage_results['status'] = 'FAIL'
            print("   ✗ DQA script not found")
        
        # Check 3.2: DQA reports
        print("\n[3.2] Verifying DQA reports...")
        dqa_reports_dir = self.reports_dir / 'dqa'
        
        if dqa_reports_dir.exists():
            report_files = list(dqa_reports_dir.glob('*.md')) + list(dqa_reports_dir.glob('*.txt'))
            if report_files:
                stage_results['checks']['dqa_reports'] = 'PASS'
                print(f"   ✓ Found {len(report_files)} DQA report(s)")
                for rf in report_files:
                    print(f"      - {rf.name}")
            else:
                stage_results['checks']['dqa_reports'] = 'FAIL'
                stage_results['issues'].append("No DQA reports found")
                print("   ✗ No DQA reports found")
        else:
            stage_results['checks']['dqa_reports'] = 'FAIL'
            stage_results['issues'].append("DQA reports directory missing")
            print("   ✗ DQA reports directory not found")
        
        # Check 3.3: DQA visualizations
        print("\n[3.3] Verifying DQA visualizations...")
        dqa_viz_dir = self.viz_dir / 'dqa'
        
        if dqa_viz_dir.exists():
            viz_files = list(dqa_viz_dir.glob('*.png')) + list(dqa_viz_dir.glob('*.jpg'))
            if viz_files:
                stage_results['checks']['dqa_visualizations'] = 'PASS'
                print(f"   ✓ Found {len(viz_files)} visualization(s)")
            else:
                stage_results['checks']['dqa_visualizations'] = 'PARTIAL'
                print("   ⚠ No DQA visualizations found")
        else:
            stage_results['checks']['dqa_visualizations'] = 'PARTIAL'
            print("   ⚠ DQA visualizations directory not found")
        
        self.verification_results['stages']['stage_3_dqa'] = stage_results
        print(f"\nStage 3 Status: {stage_results['status']}")
        return stage_results
    
    def verify_stage_4_cleaning(self):
        """Verify Stage 4: Data Cleaning"""
        print("\n" + "="*80)
        print("STAGE 4: Data Cleaning")
        print("="*80)
        
        stage_results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check 4.1: Cleaning script exists
        print("\n[4.1] Verifying data cleaning script...")
        cleaning_script = self.scripts_dir / 'data_cleaning.py'
        
        if cleaning_script.exists():
            stage_results['checks']['cleaning_script_exists'] = 'PASS'
            print(f"   ✓ Cleaning script found: {cleaning_script.name}")
            
            # Check script content
            with open(cleaning_script, 'r', encoding='utf-8') as f:
                content = f.read()
                required_ops = ['missing', 'outlier', 'duplicate', 'encoding']
                found_ops = [op for op in required_ops if op in content.lower()]
                print(f"   ✓ Found {len(found_ops)}/{len(required_ops)} cleaning operations")
        else:
            stage_results['checks']['cleaning_script_exists'] = 'FAIL'
            stage_results['issues'].append("Cleaning script missing")
            stage_results['status'] = 'FAIL'
            print("   ✗ Cleaning script not found")
        
        # Check 4.2: Cleaned data files
        print("\n[4.2] Verifying cleaned data files...")
        train_clean = self.data_dir / 'data_minihackathon_train_clean.csv'
        test_clean = self.data_dir / 'data_minihackathon_test_clean.csv'
        
        if train_clean.exists() and test_clean.exists():
            train_clean_df = pd.read_csv(train_clean)
            test_clean_df = pd.read_csv(test_clean)
            stage_results['checks']['cleaned_data_exists'] = 'PASS'
            print(f"   ✓ Cleaned train data: {train_clean_df.shape[0]} rows, {train_clean_df.shape[1]} columns")
            print(f"   ✓ Cleaned test data: {test_clean_df.shape[0]} rows, {test_clean_df.shape[1]} columns")
            
            # Verify data quality improvements
            missing_train = train_clean_df.isnull().sum().sum()
            missing_test = test_clean_df.isnull().sum().sum()
            print(f"   ✓ Missing values in train: {missing_train}")
            print(f"   ✓ Missing values in test: {missing_test}")
            
        else:
            stage_results['checks']['cleaned_data_exists'] = 'FAIL'
            stage_results['issues'].append("Cleaned data files missing")
            stage_results['status'] = 'FAIL'
            print("   ✗ Cleaned data files not found")
        
        # Check 4.3: Cleaning documentation
        print("\n[4.3] Verifying cleaning documentation...")
        cleaning_doc = self.docs_dir / 'data_cleaning_feature_engineering.md'
        
        if cleaning_doc.exists():
            stage_results['checks']['cleaning_documentation'] = 'PASS'
            print(f"   ✓ Cleaning documentation found")
        else:
            stage_results['checks']['cleaning_documentation'] = 'PARTIAL'
            print("   ⚠ Cleaning documentation not found")
        
        self.verification_results['stages']['stage_4_cleaning'] = stage_results
        print(f"\nStage 4 Status: {stage_results['status']}")
        return stage_results
    
    def verify_stage_5_feature_engineering(self):
        """Verify Stage 5: Feature Engineering"""
        print("\n" + "="*80)
        print("STAGE 5: Feature Engineering")
        print("="*80)
        
        stage_results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check 5.1: Feature engineering script exists
        print("\n[5.1] Verifying feature engineering script...")
        fe_script = self.scripts_dir / 'feature_engineering.py'
        
        if fe_script.exists():
            stage_results['checks']['fe_script_exists'] = 'PASS'
            print(f"   ✓ Feature engineering script found: {fe_script.name}")
            
            # Check script content
            with open(fe_script, 'r', encoding='utf-8') as f:
                content = f.read()
                required_features = ['interaction', 'polynomial', 'encoding', 'scaling']
                found_features = [f for f in required_features if f in content.lower()]
                print(f"   ✓ Found {len(found_features)}/{len(required_features)} feature types")
        else:
            stage_results['checks']['fe_script_exists'] = 'FAIL'
            stage_results['issues'].append("Feature engineering script missing")
            stage_results['status'] = 'FAIL'
            print("   ✗ Feature engineering script not found")
        
        # Check 5.2: Engineered data files
        print("\n[5.2] Verifying engineered data files...")
        train_eng = self.data_dir / 'data_minihackathon_train_engineered.csv'
        test_eng = self.data_dir / 'data_minihackathon_test_engineered.csv'
        
        if train_eng.exists() and test_eng.exists():
            train_eng_df = pd.read_csv(train_eng)
            test_eng_df = pd.read_csv(test_eng)
            stage_results['checks']['engineered_data_exists'] = 'PASS'
            print(f"   ✓ Engineered train data: {train_eng_df.shape[0]} rows, {train_eng_df.shape[1]} columns")
            print(f"   ✓ Engineered test data: {test_eng_df.shape[0]} rows, {test_eng_df.shape[1]} columns")
            
            # Compare with cleaned data
            train_clean = pd.read_csv(self.data_dir / 'data_minihackathon_train_clean.csv')
            new_features = train_eng_df.shape[1] - train_clean.shape[1]
            print(f"   ✓ New features created: {new_features}")
            
        else:
            stage_results['checks']['engineered_data_exists'] = 'FAIL'
            stage_results['issues'].append("Engineered data files missing")
            stage_results['status'] = 'FAIL'
            print("   ✗ Engineered data files not found")
        
        # Check 5.3: Feature engineering reports
        print("\n[5.3] Verifying feature engineering reports...")
        fe_summary = self.reports_dir / 'feature_engineering_summary.csv'
        fe_correlations = self.reports_dir / 'feature_correlations.csv'
        
        reports_found = 0
        if fe_summary.exists():
            reports_found += 1
            print(f"   ✓ Feature engineering summary found")
        if fe_correlations.exists():
            reports_found += 1
            print(f"   ✓ Feature correlations found")
        
        if reports_found > 0:
            stage_results['checks']['fe_reports'] = 'PASS' if reports_found == 2 else 'PARTIAL'
        else:
            stage_results['checks']['fe_reports'] = 'FAIL'
            stage_results['issues'].append("Feature engineering reports missing")
        
        # Check 5.4: Feature engineering visualizations
        print("\n[5.4] Verifying feature engineering visualizations...")
        fe_viz = self.viz_dir / 'feature_engineering_summary.png'
        
        if fe_viz.exists():
            stage_results['checks']['fe_visualizations'] = 'PASS'
            print(f"   ✓ Feature engineering visualization found")
        else:
            stage_results['checks']['fe_visualizations'] = 'PARTIAL'
            print("   ⚠ Feature engineering visualization not found")
        
        self.verification_results['stages']['stage_5_feature_engineering'] = stage_results
        print(f"\nStage 5 Status: {stage_results['status']}")
        return stage_results
    
    def verify_data_integrity(self):
        """Verify data integrity across pipeline"""
        print("\n" + "="*80)
        print("DATA INTEGRITY VERIFICATION")
        print("="*80)
        
        integrity_results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        try:
            # Load all data versions
            train_raw = pd.read_csv(self.data_dir / 'data_minihackathon_train.csv')
            test_raw = pd.read_csv(self.data_dir / 'data_minihackathon_test.csv')
            
            train_clean_path = self.data_dir / 'data_minihackathon_train_clean.csv'
            test_clean_path = self.data_dir / 'data_minihackathon_test_clean.csv'
            
            train_eng_path = self.data_dir / 'data_minihackathon_train_engineered.csv'
            test_eng_path = self.data_dir / 'data_minihackathon_test_engineered.csv'
            
            # Check 1: Row count consistency
            print("\n[1] Verifying row count consistency...")
            if train_clean_path.exists() and test_clean_path.exists():
                train_clean = pd.read_csv(train_clean_path)
                test_clean = pd.read_csv(test_clean_path)
                
                if train_raw.shape[0] == train_clean.shape[0]:
                    print(f"   ✓ Train rows preserved: {train_raw.shape[0]}")
                    integrity_results['checks']['train_rows_preserved'] = 'PASS'
                else:
                    print(f"   ⚠ Train rows changed: {train_raw.shape[0]} -> {train_clean.shape[0]}")
                    integrity_results['checks']['train_rows_preserved'] = 'WARN'
                
                if test_raw.shape[0] == test_clean.shape[0]:
                    print(f"   ✓ Test rows preserved: {test_raw.shape[0]}")
                    integrity_results['checks']['test_rows_preserved'] = 'PASS'
                else:
                    print(f"   ⚠ Test rows changed: {test_raw.shape[0]} -> {test_clean.shape[0]}")
                    integrity_results['checks']['test_rows_preserved'] = 'WARN'
            
            # Check 2: Column consistency between train and test
            print("\n[2] Verifying column consistency between train and test...")
            if train_eng_path.exists() and test_eng_path.exists():
                train_eng = pd.read_csv(train_eng_path)
                test_eng = pd.read_csv(test_eng_path)
                
                # Exclude target column from test
                train_cols = set(train_eng.columns) - {'target'}
                test_cols = set(test_eng.columns)
                
                if train_cols == test_cols:
                    print(f"   ✓ Feature columns match: {len(train_cols)} features")
                    integrity_results['checks']['column_consistency'] = 'PASS'
                else:
                    missing_in_test = train_cols - test_cols
                    missing_in_train = test_cols - train_cols
                    print(f"   ✗ Column mismatch detected")
                    if missing_in_test:
                        print(f"      Missing in test: {missing_in_test}")
                    if missing_in_train:
                        print(f"      Missing in train: {missing_in_train}")
                    integrity_results['checks']['column_consistency'] = 'FAIL'
                    integrity_results['issues'].append("Column mismatch between train and test")
            
            # Check 3: Data type consistency
            print("\n[3] Verifying data type consistency...")
            if train_eng_path.exists():
                train_eng = pd.read_csv(train_eng_path)
                non_numeric = train_eng.select_dtypes(exclude=[np.number]).columns.tolist()
                
                if 'id' in non_numeric:
                    non_numeric.remove('id')
                
                if len(non_numeric) == 0:
                    print(f"   ✓ All features are numeric (ready for modeling)")
                    integrity_results['checks']['data_types'] = 'PASS'
                else:
                    print(f"   ⚠ Non-numeric columns found: {non_numeric}")
                    integrity_results['checks']['data_types'] = 'WARN'
            
            # Check 4: Missing values in final data
            print("\n[4] Verifying missing values in final data...")
            if train_eng_path.exists() and test_eng_path.exists():
                train_eng = pd.read_csv(train_eng_path)
                test_eng = pd.read_csv(test_eng_path)
                
                train_missing = train_eng.isnull().sum().sum()
                test_missing = test_eng.isnull().sum().sum()
                
                if train_missing == 0 and test_missing == 0:
                    print(f"   ✓ No missing values in final data")
                    integrity_results['checks']['missing_values'] = 'PASS'
                else:
                    print(f"   ⚠ Missing values found - Train: {train_missing}, Test: {test_missing}")
                    integrity_results['checks']['missing_values'] = 'WARN'
            
        except Exception as e:
            print(f"   ✗ Error during integrity verification: {str(e)}")
            integrity_results['status'] = 'FAIL'
            integrity_results['issues'].append(f"Integrity check error: {str(e)}")
        
        self.verification_results['data_integrity'] = integrity_results
        print(f"\nData Integrity Status: {integrity_results['status']}")
        return integrity_results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY REPORT")
        print("="*80)
        
        all_checks_pass = True
        
        # Count stages
        total_stages = len(self.verification_results['stages'])
        passed_stages = sum(1 for s in self.verification_results['stages'].values() if s['status'] == 'PASS')
        
        print(f"\nStages Completed: {passed_stages}/{total_stages}")
        
        # Stage-by-stage summary
        print("\nStage-by-Stage Results:")
        for stage_name, stage_data in self.verification_results['stages'].items():
            status_symbol = "✓" if stage_data['status'] == 'PASS' else "✗"
            print(f"  {status_symbol} {stage_name.replace('_', ' ').title()}: {stage_data['status']}")
            
            if stage_data['issues']:
                for issue in stage_data['issues']:
                    print(f"      - {issue}")
            
            if stage_data['status'] != 'PASS':
                all_checks_pass = False
        
        # Data integrity summary
        if 'data_integrity' in self.verification_results:
            integrity_status = self.verification_results['data_integrity']['status']
            status_symbol = "✓" if integrity_status == 'PASS' else "⚠" if integrity_status == 'WARN' else "✗"
            print(f"\n  {status_symbol} Data Integrity: {integrity_status}")
            
            if self.verification_results['data_integrity']['issues']:
                for issue in self.verification_results['data_integrity']['issues']:
                    print(f"      - {issue}")
        
        # Overall status
        if all_checks_pass and passed_stages == total_stages:
            self.verification_results['overall_status'] = 'PASS'
        elif passed_stages >= total_stages * 0.8:
            self.verification_results['overall_status'] = 'PARTIAL'
        else:
            self.verification_results['overall_status'] = 'FAIL'
        
        print("\n" + "="*80)
        print(f"OVERALL STATUS: {self.verification_results['overall_status']}")
        print("="*80)
        
        # Recommendations
        print("\nRecommendations:")
        if self.verification_results['overall_status'] == 'PASS':
            print("  ✓ All verification checks passed!")
            print("  ✓ Pipeline is ready for model training")
            self.verification_results['recommendations'].append("Ready for modeling stage")
        else:
            print("  ⚠ Some issues found - review above details")
            if passed_stages < total_stages:
                print(f"  ⚠ Complete remaining {total_stages - passed_stages} stage(s)")
                self.verification_results['recommendations'].append("Complete remaining stages")
            
            # Collect all issues
            all_issues = []
            for stage_data in self.verification_results['stages'].values():
                all_issues.extend(stage_data['issues'])
            
            if 'data_integrity' in self.verification_results:
                all_issues.extend(self.verification_results['data_integrity']['issues'])
            
            if all_issues:
                print("\n  Issues to Address:")
                for idx, issue in enumerate(all_issues, 1):
                    print(f"    {idx}. {issue}")
                    if issue not in self.verification_results['recommendations']:
                        self.verification_results['recommendations'].append(f"Fix: {issue}")
        
        return self.verification_results
    
    def save_verification_report(self, output_path=None):
        """Save verification report to file"""
        if output_path is None:
            output_path = self.base_dir / 'VERIFICATION_REPORT.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Pipeline Verification Report\n\n")
            f.write(f"**Generated:** {self.verification_results['timestamp']}\n\n")
            f.write(f"**Overall Status:** {self.verification_results['overall_status']}\n\n")
            
            f.write("## Stage Results\n\n")
            for stage_name, stage_data in self.verification_results['stages'].items():
                f.write(f"### {stage_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Status:** {stage_data['status']}\n\n")
                
                f.write("**Checks:**\n")
                for check_name, check_result in stage_data['checks'].items():
                    f.write(f"- {check_name}: {check_result}\n")
                
                if stage_data['issues']:
                    f.write("\n**Issues:**\n")
                    for issue in stage_data['issues']:
                        f.write(f"- {issue}\n")
                
                f.write("\n")
            
            if 'data_integrity' in self.verification_results:
                f.write("## Data Integrity\n\n")
                integrity = self.verification_results['data_integrity']
                f.write(f"**Status:** {integrity['status']}\n\n")
                
                f.write("**Checks:**\n")
                for check_name, check_result in integrity['checks'].items():
                    f.write(f"- {check_name}: {check_result}\n")
                
                if integrity['issues']:
                    f.write("\n**Issues:**\n")
                    for issue in integrity['issues']:
                        f.write(f"- {issue}\n")
            
            f.write("\n## Recommendations\n\n")
            for rec in self.verification_results['recommendations']:
                f.write(f"- {rec}\n")
        
        print(f"\nVerification report saved to: {output_path}")
        return output_path
    
    def run_full_verification(self):
        """Run complete verification pipeline"""
        print("\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + " "*20 + "PIPELINE VERIFICATION AGENT" + " "*32 + "#")
        print("#" + " "*78 + "#")
        print("#"*80)
        
        # Run all verification stages
        self.verify_stage_1_exploration()
        self.verify_stage_2_eda()
        self.verify_stage_3_dqa()
        self.verify_stage_4_cleaning()
        self.verify_stage_5_feature_engineering()
        self.verify_data_integrity()
        
        # Generate summary
        results = self.generate_summary_report()
        
        # Save report
        report_path = self.save_verification_report()
        
        return results


def main():
    """Main execution function"""
    print("\nInitializing Verification Agent...")
    agent = VerificationAgent()
    
    print("Starting full pipeline verification...\n")
    results = agent.run_full_verification()
    
    # Save results as JSON for programmatic access
    json_path = agent.base_dir / 'verification_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")
    
    return results


if __name__ == "__main__":
    results = main()
