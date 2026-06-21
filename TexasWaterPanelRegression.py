"""
Texas Water Usage Panel Regression Analysis
Implements difference-in-differences models to analyze the impact of
hyperscaler and AI data centers on water usage in Texas counties.

Models include:
1. Basic DiD with hyperscaler treatment post-2021
2. DiD with AI datacenter treatment
3. Capacity-weighted treatment effects
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels import PanelOLS
from linearmodels.panel import compare
import os
import warnings
warnings.filterwarnings('ignore')

# Set working directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
CODE_DIR = os.path.join(BASE_DIR, 'Code')

print("="*70)
print("TEXAS WATER USAGE - PANEL REGRESSION ANALYSIS")
print("="*70)

def load_and_prepare_data():
    """Load and prepare panel data for regression analysis"""
    print("\n1. Loading and Preparing Data...")
    print("-"*50)
    
    os.chdir(CODE_DIR)
    
    # Load main analysis data
    df = pd.read_csv('texas_water_datacenter_analysis.csv')
    
    # Load datacenter details for hyperscaler identification
    dc_df = pd.read_csv('texas_datacenters_with_counties.csv')
    
    # Define hyperscaler companies
    hyperscalers = ['Amazon', 'Microsoft', 'Google', 'Meta', 'Facebook', 'Apple', 'Oracle']
    hyperscaler_pattern = '|'.join(hyperscalers)
    dc_df['is_hyperscaler'] = dc_df['PROVIDER_NAME'].str.contains(
        hyperscaler_pattern, case=False, na=False
    )
    
    # Get counties with hyperscalers
    hyperscaler_counties = dc_df[dc_df['is_hyperscaler']]['CNTY_NM'].str.upper().unique()
    
    # Add hyperscaler flag to main dataframe
    df['has_hyperscaler'] = df['CountyName'].str.upper().isin(hyperscaler_counties).astype(int)
    
    print(f"Total observations: {len(df)}")
    print(f"Counties: {df['CountyName'].nunique()}")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Counties with hyperscalers: {df.groupby('CountyName')['has_hyperscaler'].max().sum()}")
    print(f"Counties with AI datacenters: {df.groupby('CountyName')['has_ai_datacenter'].max().sum()}")
    
    return df

def create_panel_variables(df):
    """Create variables for panel regression"""
    print("\n2. Creating Panel Variables...")
    print("-"*50)
    
    # Calculate total water usage and per capita measures
    water_categories = ['Municipal', 'Manufacturing', 'Mining', 'Power', 'Irrigation', 'LiveStock']
    df['Total_Water'] = df[water_categories].sum(axis=1)
    
    # Per capita measures (gallons per person)
    df['Water_PerCapita'] = (df['Total_Water'] * 1000000) / df['Population']
    df['Power_PerCapita'] = (df['Power'] * 1000000) / df['Population']
    df['Municipal_PerCapita'] = (df['Municipal'] * 1000000) / df['Population']
    
    # Log transformations (adding small constant to handle zeros)
    epsilon = 1  # Small constant to avoid log(0)
    df['log_Water_PerCapita'] = np.log(df['Water_PerCapita'] + epsilon)
    df['log_Power_PerCapita'] = np.log(df['Power_PerCapita'] + epsilon)
    df['log_Municipal_PerCapita'] = np.log(df['Municipal_PerCapita'] + epsilon)
    df['log_Population'] = np.log(df['Population'])
    
    # Treatment variables
    df['post_2021'] = (df['Year'] >= 2021).astype(int)
    df['hyperscaler_post2021'] = df['has_hyperscaler'] * df['post_2021']
    df['ai_datacenter_post2021'] = df['has_ai_datacenter'] * df['post_2021']
    
    # Capacity interaction (in GW for better scaling)
    df['ai_capacity_gw'] = df['ai_datacenter_capacity_mw'] / 1000
    df['ai_capacity_post2021'] = df['ai_capacity_gw'] * df['post_2021']
    
    # Time trend
    df['time_trend'] = df['Year'] - df['Year'].min()
    
    # Create lagged dependent variable for AR(1) term
    df = df.sort_values(['CountyName', 'Year'])
    df['lag_log_Water_PerCapita'] = df.groupby('CountyName')['log_Water_PerCapita'].shift(1)
    df['lag_log_Power_PerCapita'] = df.groupby('CountyName')['log_Power_PerCapita'].shift(1)
    df['lag_log_Municipal_PerCapita'] = df.groupby('CountyName')['log_Municipal_PerCapita'].shift(1)
    
    # Create panel index
    df['county_id'] = pd.Categorical(df['CountyName']).codes
    df = df.set_index(['county_id', 'Year'])
    
    # Remove observations with missing lagged values (first year)
    df = df.dropna(subset=['lag_log_Water_PerCapita'])
    
    print(f"Observations after creating lags: {len(df)}")
    print(f"Treatment groups:")
    print(f"  Hyperscaler treated (post-2021): {(df['hyperscaler_post2021'] == 1).sum()}")
    print(f"  AI datacenter treated (post-2021): {(df['ai_datacenter_post2021'] == 1).sum()}")
    print(f"  Counties with positive AI capacity: {(df['ai_capacity_gw'] > 0).sum()}")
    
    return df

def run_hyperscaler_regressions(df):
    """Run DiD regressions for hyperscaler treatment"""
    print("\n3. Running Hyperscaler Treatment Regressions...")
    print("-"*50)
    
    results = {}
    
    # Dependent variables to analyze
    dep_vars = {
        'Total Water': 'log_Water_PerCapita',
        'Power': 'log_Power_PerCapita',
        'Municipal': 'log_Municipal_PerCapita'
    }
    
    for name, dep_var in dep_vars.items():
        print(f"\nRegression for {name}:")
        
        # Prepare data
        lag_var = f'lag_{dep_var}'
        
        # Model specification - remove time-invariant treatment dummy with fixed effects
        exog_vars = ['time_trend', lag_var, 'post_2021', 'hyperscaler_post2021']
        
        # Remove any remaining NaN values
        reg_data = df[[dep_var] + exog_vars].dropna()
        
        # Run panel regression with fixed effects (no constant needed with entity effects)
        model = PanelOLS(
            reg_data[dep_var], 
            reg_data[exog_vars],
            entity_effects=True,
            time_effects=False,
            drop_absorbed=True
        )
        
        result = model.fit(cov_type='clustered', cluster_entity=True)
        results[name] = result
        
        # Print key results
        print(f"  DiD coefficient (hyperscaler_post2021): {result.params['hyperscaler_post2021']:.4f}")
        print(f"  Standard error: {result.std_errors['hyperscaler_post2021']:.4f}")
        print(f"  P-value: {result.pvalues['hyperscaler_post2021']:.4f}")
        print(f"  R-squared: {result.rsquared:.4f}")
        print(f"  N observations: {result.nobs}")
    
    return results

def run_ai_datacenter_regressions(df):
    """Run DiD regressions for AI datacenter treatment"""
    print("\n4. Running AI Datacenter Treatment Regressions...")
    print("-"*50)
    
    results = {}
    
    dep_vars = {
        'Total Water': 'log_Water_PerCapita',
        'Power': 'log_Power_PerCapita',
        'Municipal': 'log_Municipal_PerCapita'
    }
    
    for name, dep_var in dep_vars.items():
        print(f"\nRegression for {name}:")
        
        # Prepare data
        lag_var = f'lag_{dep_var}'
        
        # Model specification - remove time-invariant treatment dummy with fixed effects
        exog_vars = ['time_trend', lag_var, 'post_2021', 'ai_datacenter_post2021']
        
        # Remove any remaining NaN values
        reg_data = df[[dep_var] + exog_vars].dropna()
        
        # Run panel regression with fixed effects (no constant needed with entity effects)
        model = PanelOLS(
            reg_data[dep_var], 
            reg_data[exog_vars],
            entity_effects=True,
            time_effects=False,
            drop_absorbed=True
        )
        
        result = model.fit(cov_type='clustered', cluster_entity=True)
        results[name] = result
        
        # Print key results
        print(f"  DiD coefficient (ai_datacenter_post2021): {result.params['ai_datacenter_post2021']:.4f}")
        print(f"  Standard error: {result.std_errors['ai_datacenter_post2021']:.4f}")
        print(f"  P-value: {result.pvalues['ai_datacenter_post2021']:.4f}")
        print(f"  R-squared: {result.rsquared:.4f}")
        print(f"  N observations: {result.nobs}")
    
    return results

def run_capacity_weighted_regressions(df):
    """Run regressions with capacity-weighted treatment"""
    print("\n5. Running Capacity-Weighted Treatment Regressions...")
    print("-"*50)
    
    results = {}
    
    dep_vars = {
        'Total Water': 'log_Water_PerCapita',
        'Power': 'log_Power_PerCapita',
        'Municipal': 'log_Municipal_PerCapita'
    }
    
    for name, dep_var in dep_vars.items():
        print(f"\nRegression for {name}:")
        
        # Prepare data
        lag_var = f'lag_{dep_var}'
        
        # Model specification with capacity interaction
        exog_vars = ['time_trend', lag_var, 'post_2021', 'ai_capacity_post2021']
        
        # Remove any remaining NaN values
        reg_data = df[[dep_var] + exog_vars].dropna()
        
        # Run panel regression with fixed effects (no constant needed with entity effects)
        model = PanelOLS(
            reg_data[dep_var], 
            reg_data[exog_vars],
            entity_effects=True,
            time_effects=False,
            drop_absorbed=True
        )
        
        result = model.fit(cov_type='clustered', cluster_entity=True)
        results[name] = result
        
        # Print key results
        print(f"  Capacity interaction coefficient: {result.params['ai_capacity_post2021']:.4f}")
        print(f"  Standard error: {result.std_errors['ai_capacity_post2021']:.4f}")
        print(f"  P-value: {result.pvalues['ai_capacity_post2021']:.4f}")
        print(f"  R-squared: {result.rsquared:.4f}")
        print(f"  N observations: {result.nobs}")
    
    return results

def create_regression_tables(hyperscaler_results, ai_results, capacity_results):
    """Create formatted regression tables"""
    print("\n6. Creating Regression Tables...")
    print("-"*50)
    
    # Create summary table for all models
    all_results = []
    model_names = []
    
    # Add hyperscaler models
    for name, result in hyperscaler_results.items():
        all_results.append(result)
        model_names.append(f"Hyperscaler\n{name}")
    
    # Add AI datacenter models
    for name, result in ai_results.items():
        all_results.append(result)
        model_names.append(f"AI DC\n{name}")
    
    # Add capacity-weighted models
    for name, result in capacity_results.items():
        all_results.append(result)
        model_names.append(f"Capacity\n{name}")
    
    # Create comparison table
    summary = compare(all_results, stars=True)
    
    # Save to text file
    with open('texas_water_regression_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEXAS WATER USAGE - PANEL REGRESSION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("All models include:\n")
        f.write("- County fixed effects\n")
        f.write("- Time trend\n")
        f.write("- AR(1) term (lagged dependent variable)\n")
        f.write("- Clustered standard errors at county level\n")
        f.write("- Treatment period: Post-2021\n\n")
        f.write(str(summary))
    
    print("Regression results saved to: texas_water_regression_results.txt")
    
    return summary

def create_detailed_results_table(hyperscaler_results, ai_results, capacity_results):
    """Create a more detailed results table in CSV format"""
    print("\n7. Creating Detailed Results Table...")
    print("-"*50)
    
    results_data = []
    
    # Extract results for hyperscaler models
    for dep_name, result in hyperscaler_results.items():
        treatment_coef = result.params.get('hyperscaler_post2021', np.nan)
        treatment_se = result.std_errors.get('hyperscaler_post2021', np.nan)
        treatment_pval = result.pvalues.get('hyperscaler_post2021', np.nan)
        
        results_data.append({
            'Model': 'Hyperscaler DiD',
            'Dependent Variable': dep_name,
            'Treatment Coefficient': treatment_coef,
            'Standard Error': treatment_se,
            'P-value': treatment_pval,
            'Significant at 5%': 'Yes' if treatment_pval < 0.05 else 'No',
            'Significant at 10%': 'Yes' if treatment_pval < 0.10 else 'No',
            'R-squared': result.rsquared,
            'N Observations': result.nobs,
            'N Counties': result.entity_info['total'],
            'Effect Size (%)': (np.exp(treatment_coef) - 1) * 100 if not np.isnan(treatment_coef) else np.nan
        })
    
    # Extract results for AI datacenter models
    for dep_name, result in ai_results.items():
        treatment_coef = result.params.get('ai_datacenter_post2021', np.nan)
        treatment_se = result.std_errors.get('ai_datacenter_post2021', np.nan)
        treatment_pval = result.pvalues.get('ai_datacenter_post2021', np.nan)
        
        results_data.append({
            'Model': 'AI Datacenter DiD',
            'Dependent Variable': dep_name,
            'Treatment Coefficient': treatment_coef,
            'Standard Error': treatment_se,
            'P-value': treatment_pval,
            'Significant at 5%': 'Yes' if treatment_pval < 0.05 else 'No',
            'Significant at 10%': 'Yes' if treatment_pval < 0.10 else 'No',
            'R-squared': result.rsquared,
            'N Observations': result.nobs,
            'N Counties': result.entity_info['total'],
            'Effect Size (%)': (np.exp(treatment_coef) - 1) * 100 if not np.isnan(treatment_coef) else np.nan
        })
    
    # Extract results for capacity-weighted models
    for dep_name, result in capacity_results.items():
        treatment_coef = result.params.get('ai_capacity_post2021', np.nan)
        treatment_se = result.std_errors.get('ai_capacity_post2021', np.nan)
        treatment_pval = result.pvalues.get('ai_capacity_post2021', np.nan)
        
        results_data.append({
            'Model': 'Capacity-Weighted',
            'Dependent Variable': dep_name,
            'Treatment Coefficient': treatment_coef,
            'Standard Error': treatment_se,
            'P-value': treatment_pval,
            'Significant at 5%': 'Yes' if treatment_pval < 0.05 else 'No',
            'Significant at 10%': 'Yes' if treatment_pval < 0.10 else 'No',
            'R-squared': result.rsquared,
            'N Observations': result.nobs,
            'N Counties': result.entity_info['total'],
            'Effect Size (per GW)': treatment_coef  # Already in log scale
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    results_df.to_csv('texas_water_regression_summary.csv', index=False)
    print("Detailed results saved to: texas_water_regression_summary.csv")
    
    # Print summary table
    print("\n" + "="*80)
    print("REGRESSION RESULTS SUMMARY")
    print("="*80)
    
    for model in results_df['Model'].unique():
        print(f"\n{model}:")
        model_results = results_df[results_df['Model'] == model]
        for _, row in model_results.iterrows():
            print(f"  {row['Dependent Variable']}:")
            print(f"    Coefficient: {row['Treatment Coefficient']:.4f}")
            print(f"    P-value: {row['P-value']:.4f}")
            if 'Effect Size (%)' in row and not pd.isna(row.get('Effect Size (%)')):
                print(f"    Effect size: {row['Effect Size (%)']:.2f}%")
            print(f"    Significant at 5%: {row['Significant at 5%']}")
    
    return results_df

def run_robustness_checks(df):
    """Run robustness checks with alternative specifications"""
    print("\n8. Running Robustness Checks...")
    print("-"*50)
    
    # 1. Model with both hyperscaler and AI datacenter treatments
    print("\nRobustness Check 1: Combined Treatment Model")
    
    exog_vars = ['time_trend', 'lag_log_Water_PerCapita', 'post_2021',
                 'hyperscaler_post2021', 'ai_datacenter_post2021']
    
    reg_data = df[['log_Water_PerCapita'] + exog_vars].dropna()
    
    model = PanelOLS(
        reg_data['log_Water_PerCapita'],
        reg_data[exog_vars],
        entity_effects=True,
        drop_absorbed=True
    )
    
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    print(f"  Hyperscaler effect: {result.params['hyperscaler_post2021']:.4f} (p={result.pvalues['hyperscaler_post2021']:.4f})")
    print(f"  AI datacenter effect: {result.params['ai_datacenter_post2021']:.4f} (p={result.pvalues['ai_datacenter_post2021']:.4f})")
    
    # 2. Model with year fixed effects instead of time trend
    print("\nRobustness Check 2: Year Fixed Effects Model")
    
    exog_vars = ['lag_log_Water_PerCapita', 'ai_datacenter_post2021']
    
    reg_data = df[['log_Water_PerCapita'] + exog_vars].dropna()
    
    model = PanelOLS(
        reg_data['log_Water_PerCapita'],
        reg_data[exog_vars],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    print(f"  AI datacenter effect: {result.params['ai_datacenter_post2021']:.4f} (p={result.pvalues['ai_datacenter_post2021']:.4f})")
    
    # 3. Placebo test with fake treatment year (2018)
    print("\nRobustness Check 3: Placebo Test (Treatment in 2018)")
    
    df_placebo = df.copy()
    df_placebo['post_2018'] = (df_placebo.index.get_level_values('Year') >= 2018).astype(int)
    df_placebo['ai_datacenter_post2018'] = df_placebo['has_ai_datacenter'] * df_placebo['post_2018']
    
    # Only use pre-2021 data for placebo test
    df_placebo = df_placebo[df_placebo.index.get_level_values('Year') < 2021]
    
    exog_vars = ['time_trend', 'lag_log_Water_PerCapita', 'post_2018', 'ai_datacenter_post2018']
    
    reg_data = df_placebo[['log_Water_PerCapita'] + exog_vars].dropna()
    
    model = PanelOLS(
        reg_data['log_Water_PerCapita'],
        reg_data[exog_vars],
        entity_effects=True,
        drop_absorbed=True
    )
    
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    print(f"  Placebo effect (2018): {result.params['ai_datacenter_post2018']:.4f} (p={result.pvalues['ai_datacenter_post2018']:.4f})")
    print(f"  Interpretation: Should be insignificant if 2021 is the true treatment year")

def create_pre_trends_analysis(df):
    """Analyze pre-treatment trends"""
    print("\n9. Pre-Treatment Trends Analysis...")
    print("-"*50)
    
    # Reset index for easier manipulation
    df_trends = df.reset_index()
    
    # Focus on pre-treatment period (before 2021)
    pre_treatment = df_trends[df_trends['Year'] < 2021].copy()
    
    # Calculate average annual growth rates by group
    def calculate_growth_rate(group):
        if len(group) < 2:
            return np.nan
        # Log difference approximates growth rate
        return group['log_Water_PerCapita'].diff().mean()
    
    # For hyperscaler counties
    hyperscaler_growth = pre_treatment.groupby('has_hyperscaler').apply(calculate_growth_rate)
    print(f"Pre-treatment annual growth (2000-2020):")
    print(f"  Non-hyperscaler counties: {hyperscaler_growth[0]:.4f}")
    print(f"  Hyperscaler counties: {hyperscaler_growth[1]:.4f}")
    print(f"  Difference: {hyperscaler_growth[1] - hyperscaler_growth[0]:.4f}")
    
    # For AI datacenter counties
    ai_growth = pre_treatment.groupby('has_ai_datacenter').apply(calculate_growth_rate)
    print(f"\n  Non-AI datacenter counties: {ai_growth[0]:.4f}")
    print(f"  AI datacenter counties: {ai_growth[1]:.4f}")
    print(f"  Difference: {ai_growth[1] - ai_growth[0]:.4f}")
    
    # Test for parallel trends using interaction with pre-treatment time
    print("\nFormal parallel trends test:")
    
    pre_treatment['time_x_ai'] = pre_treatment['time_trend'] * pre_treatment['has_ai_datacenter']
    
    # Set index for panel regression
    pre_treatment = pre_treatment.set_index(['county_id', 'Year'])
    
    exog_vars = ['time_trend', 'time_x_ai']
    reg_data = pre_treatment[['log_Water_PerCapita'] + exog_vars].dropna()
    
    model = PanelOLS(
        reg_data['log_Water_PerCapita'],
        reg_data[exog_vars],
        entity_effects=True,
        drop_absorbed=True
    )
    
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    print(f"  Time × AI datacenter interaction: {result.params['time_x_ai']:.6f}")
    print(f"  P-value: {result.pvalues['time_x_ai']:.4f}")
    print(f"  Interpretation: P>0.10 suggests parallel trends assumption holds")

def main():
    """Main execution function"""
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Create panel variables
        df = create_panel_variables(df)
        
        # Run main regressions
        hyperscaler_results = run_hyperscaler_regressions(df)
        ai_results = run_ai_datacenter_regressions(df)
        capacity_results = run_capacity_weighted_regressions(df)
        
        # Create output tables
        summary = create_regression_tables(hyperscaler_results, ai_results, capacity_results)
        results_df = create_detailed_results_table(hyperscaler_results, ai_results, capacity_results)
        
        # Run robustness checks
        run_robustness_checks(df)
        
        # Analyze pre-trends
        create_pre_trends_analysis(df)
        
        print("\n" + "="*70)
        print("PANEL REGRESSION ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("1. texas_water_regression_results.txt - Full regression output")
        print("2. texas_water_regression_summary.csv - Summary table")
        print("\nKey findings displayed above")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
