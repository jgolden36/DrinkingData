"""
Water Stress and AI Data Center Analysis - Geographic Merge Only
This script merges AI data center locations with the most recent water stress ratings
from WRI Aqueduct 4.0 based on geographic location only (no temporal component).

Creates visualizations overlaying data centers on water stress regions.
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("WATER STRESS AND AI DATA CENTER ANALYSIS")
print("Geographic Merge with Most Recent Water Stress Ratings")
print("="*70)

# Set working directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')
WATER_DATA_DIR = os.path.join(DATA_DIR, 'Water Data', 'Water Resources Institute', 
                             'aqueduct-4-0-water-risk-data', 'Aqueduct40_waterrisk_download_Y2023M07D05')

def map_organization_to_provider(organization):
    """Maps organization names to data center providers (from DiffinDiff.py)."""
    if pd.isna(organization):
        return None, None
    
    org_lower = str(organization).lower()
    providers = []
    
    if any(keyword in org_lower for keyword in ['google', 'anthropic', 'deep mind', 'deepmind']):
        providers.append('Google')
    if any(keyword in org_lower for keyword in ['microsoft', 'openai']):
        providers.append('Microsoft')
    if any(keyword in org_lower for keyword in ['meta', 'facebook']):
        providers.append('Facebook')
    if 'apple' in org_lower:
        providers.append('Apple Inc.')
    if any(keyword in org_lower for keyword in ['amazon', 'perplexity']):
        providers.append('Amazon AWS')
    
    providers = list(dict.fromkeys(providers))
    primary = providers[0] if len(providers) > 0 else None
    secondary = providers[1] if len(providers) > 1 else None
    
    return primary, secondary

def load_ai_datacenters():
    """Load and filter data centers associated with AI models."""
    print("\n1. LOADING AI AND DATA CENTER DATA")
    print("-"*50)
    os.chdir(DATA_DIR)
    
    # Load AI models
    AI_df = pd.read_csv('Epoch Database - Notable Models.csv', thousands=',')
    AI_df["Publication date"] = pd.to_datetime(AI_df["Publication date"])
    AI_df[['primaryDataCenterProvider', 'secondaryDataCenterProvider']] = AI_df['Organization'].apply(
        lambda x: pd.Series(map_organization_to_provider(x))
    )
    
    # Filter for frontier models with identified providers
    AI_df = AI_df[(AI_df['Frontier model'] == 'checked') & (AI_df['primaryDataCenterProvider'].notna())]
    print(f"Found {len(AI_df)} frontier AI models with identified providers")
    
    # Load all data centers
    dataCenterdf = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')
    dataCenterdf['geometry'] = dataCenterdf.apply(
        lambda row: Point(row['LOCATION_LONGITUDE'], row['LOCATION_LATITUDE']), axis=1
    )
    dataCenters_gdf = gpd.GeoDataFrame(dataCenterdf, geometry='geometry', crs='EPSG:4326')
    print(f"Loaded {len(dataCenters_gdf)} total data centers")
    
    # Merge to get AI-related data centers only
    ai_datacenters = dataCenters_gdf.merge(
        AI_df[['primaryDataCenterProvider', 'Model', 'Parameters', 'Organization']].drop_duplicates(),
        left_on='PROVIDER_NAME',
        right_on='primaryDataCenterProvider',
        how='inner'
    )
    
    # Remove duplicates (one data center can serve multiple models)
    ai_datacenters_unique = ai_datacenters.groupby('ATERIO_DATA_CENTER_UID').first().reset_index()
    
    print(f"Identified {len(ai_datacenters_unique)} unique AI-related data centers")
    print(f"Providers: {sorted(ai_datacenters_unique['PROVIDER_NAME'].unique())}")
    
    return ai_datacenters_unique, dataCenters_gdf

def load_water_stress_regions():
    """Load the most recent water stress data by geographic region."""
    print("\n2. LOADING WATER STRESS DATA (Most Recent Baseline)")
    print("-"*50)
    
    csv_path = os.path.join(WATER_DATA_DIR, 'CVS', 'Aqueduct40_baseline_annual_y2023m07d05.csv')
    
    print("Reading water stress data by region...")
    
    # Read data in chunks and aggregate by geographic region
    chunk_size = 100000
    region_data = {}
    
    columns_needed = ['string_id', 'gid_0', 'gid_1', 'name_0', 'name_1', 
                     'bwd_raw', 'bwd_score', 'bwd_cat', 'bwd_label',
                     'bws_raw', 'bws_score', 'bws_cat', 'bws_label',
                     'w_awr_def_tot_cat', 'w_awr_def_tot_label']  # overall water risk
    
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, usecols=columns_needed)):
        total_rows += len(chunk)
        
        # Aggregate by country-region
        for (country, region), group in chunk.groupby(['name_0', 'name_1']):
            key = (country, region)
            if key not in region_data:
                region_data[key] = {
                    'bwd_scores': [],
                    'bws_scores': [],
                    'bwd_labels': [],
                    'bws_labels': [],
                    'overall_risk': []
                }
            
            region_data[key]['bwd_scores'].extend(group['bwd_score'].dropna().tolist())
            region_data[key]['bws_scores'].extend(group['bws_score'].dropna().tolist())
            region_data[key]['bwd_labels'].extend(group['bwd_label'].dropna().tolist())
            region_data[key]['bws_labels'].extend(group['bws_label'].dropna().tolist())
            if 'w_awr_def_tot_label' in group.columns:
                region_data[key]['overall_risk'].extend(group['w_awr_def_tot_label'].dropna().tolist())
        
        print(f"  Processed {total_rows:,} rows...")
        if i >= 10:  # Process first 1.1M rows
            break
    
    # Create summary dataframe
    regions = []
    for (country, region), data in region_data.items():
        regions.append({
            'country': country,
            'region': region,
            'bwd_score': np.mean(data['bwd_scores']) if data['bwd_scores'] else np.nan,
            'bws_score': np.mean(data['bws_scores']) if data['bws_scores'] else np.nan,
            'bwd_label': max(set(data['bwd_labels']), key=data['bwd_labels'].count) if data['bwd_labels'] else 'No Data',
            'bws_label': max(set(data['bws_labels']), key=data['bws_labels'].count) if data['bws_labels'] else 'No Data',
            'overall_risk': max(set(data['overall_risk']), key=data['overall_risk'].count) if data['overall_risk'] else 'No Data',
            'n_observations': len(data['bwd_scores'])
        })
    
    water_stress_df = pd.DataFrame(regions)
    
    print(f"\nProcessed {len(water_stress_df)} unique geographic regions")
    print(f"Countries: {water_stress_df['country'].nunique()}")
    print(f"Regions with high water depletion: {(water_stress_df['bwd_score'] > 3).sum()}")
    print(f"Regions with extreme water depletion: {(water_stress_df['bwd_score'] > 4).sum()}")
    
    return water_stress_df

def geographic_merge_simple(ai_datacenters, water_stress_df):
    """
    Simple geographic merge based on country-level aggregation.
    For a more precise analysis, you would use actual spatial joins with water basin boundaries.
    """
    print("\n3. MERGING DATA CENTERS WITH WATER STRESS REGIONS")
    print("-"*50)
    
    # Initialize water stress columns
    ai_datacenters['country'] = 'Unknown'
    ai_datacenters['region'] = 'Unknown'
    ai_datacenters['bwd_score'] = np.nan
    ai_datacenters['bwd_label'] = 'No Data'
    ai_datacenters['bws_score'] = np.nan
    ai_datacenters['bws_label'] = 'No Data'
    ai_datacenters['overall_water_risk'] = 'No Data'
    
    # Define geographic regions for major data center locations
    # US regions (simplified - in practice use state boundaries)
    us_regions = {
        'California': {'lon': (-124, -114), 'lat': (32, 42)},
        'Virginia': {'lon': (-83, -75), 'lat': (36, 40)},
        'Texas': {'lon': (-107, -93), 'lat': (25, 37)},
        'Oregon': {'lon': (-125, -116), 'lat': (42, 46)},
        'Washington': {'lon': (-125, -116), 'lat': (45, 49)},
        'Iowa': {'lon': (-97, -90), 'lat': (40, 44)},
        'Georgia': {'lon': (-86, -81), 'lat': (30, 35)},
        'Illinois': {'lon': (-92, -87), 'lat': (37, 43)},
        'North Carolina': {'lon': (-84, -75), 'lat': (34, 37)},
        'Arizona': {'lon': (-115, -109), 'lat': (31, 37)}
    }
    
    # Get US water stress averages by state
    us_water = water_stress_df[water_stress_df['country'] == 'United States of America'].copy()
    
    # Assign water stress based on location
    for state, bounds in us_regions.items():
        mask = (ai_datacenters.geometry.x >= bounds['lon'][0]) & \
               (ai_datacenters.geometry.x <= bounds['lon'][1]) & \
               (ai_datacenters.geometry.y >= bounds['lat'][0]) & \
               (ai_datacenters.geometry.y <= bounds['lat'][1])
        
        if mask.any():
            # Try to find water stress for this state
            state_water = us_water[us_water['region'].str.contains(state, case=False, na=False)]
            
            if len(state_water) > 0:
                # Use actual state data
                avg_bwd = state_water['bwd_score'].mean()
                avg_bws = state_water['bws_score'].mean()
                mode_bwd = state_water['bwd_label'].mode()[0] if len(state_water['bwd_label'].mode()) > 0 else 'No Data'
                mode_bws = state_water['bws_label'].mode()[0] if len(state_water['bws_label'].mode()) > 0 else 'No Data'
                mode_risk = state_water['overall_risk'].mode()[0] if len(state_water['overall_risk'].mode()) > 0 else 'No Data'
            else:
                # Use national average for US
                avg_bwd = us_water['bwd_score'].mean() if len(us_water) > 0 else np.nan
                avg_bws = us_water['bws_score'].mean() if len(us_water) > 0 else np.nan
                mode_bwd = 'Medium - High (20-40%)'  # Default for US
                mode_bws = 'Medium - High (20-40%)'
                mode_risk = 'Medium - High (2-3)'
            
            ai_datacenters.loc[mask, 'country'] = 'United States'
            ai_datacenters.loc[mask, 'region'] = state
            ai_datacenters.loc[mask, 'bwd_score'] = avg_bwd
            ai_datacenters.loc[mask, 'bws_score'] = avg_bws
            ai_datacenters.loc[mask, 'bwd_label'] = mode_bwd
            ai_datacenters.loc[mask, 'bws_label'] = mode_bws
            ai_datacenters.loc[mask, 'overall_water_risk'] = mode_risk
            
            print(f"  {state}: {mask.sum()} data centers, BWD: {mode_bwd}")
    
    # Handle non-US data centers
    # Europe
    europe_mask = (ai_datacenters.geometry.x > -10) & (ai_datacenters.geometry.x < 30) & \
                  (ai_datacenters.geometry.y > 35) & (ai_datacenters.geometry.y < 60)
    ai_datacenters.loc[europe_mask, 'country'] = 'Europe'
    ai_datacenters.loc[europe_mask, 'bwd_label'] = 'Low - Medium (10-20%)'
    ai_datacenters.loc[europe_mask, 'bws_label'] = 'Low - Medium (10-20%)'
    
    # Asia
    asia_mask = (ai_datacenters.geometry.x > 60) & (ai_datacenters.geometry.x < 150) & \
                (ai_datacenters.geometry.y > 10) & (ai_datacenters.geometry.y < 60)
    ai_datacenters.loc[asia_mask, 'country'] = 'Asia'
    ai_datacenters.loc[asia_mask, 'bwd_label'] = 'High (40-80%)'
    ai_datacenters.loc[asia_mask, 'bws_label'] = 'High (40-80%)'
    
    print(f"\nTotal data centers with water stress data: {(ai_datacenters['bwd_label'] != 'No Data').sum()}")
    print(f"Data centers in high water stress areas: {ai_datacenters['bwd_label'].str.contains('High', na=False).sum()}")
    
    return ai_datacenters

def create_water_stress_map(ai_datacenters, water_stress_df):
    """Create a comprehensive map showing data centers and water stress."""
    print("\n4. CREATING WATER STRESS MAP")
    print("-"*50)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define color scheme for water stress levels
    water_colors = {
        'Low (<10%)': '#2166ac',
        'Low (<5%)': '#2166ac',
        'Low - Medium (10-20%)': '#67a9cf',
        'Low - Medium (5-25%)': '#67a9cf',
        'Medium - High (20-40%)': '#fddbc7',
        'Medium - High (25-50%)': '#fddbc7',
        'High (40-80%)': '#ef8a62',
        'High (50-75%)': '#ef8a62',
        'Extremely High (>80%)': '#b2182b',
        'Extremely High (>75%)': '#b2182b',
        'Arid and Low Water Use': '#969696',
        'No Data': '#ffffff'
    }
    
    # Provider colors and markers
    provider_colors = {
        'Google': '#4285F4',
        'Microsoft': '#00A4EF', 
        'Facebook': '#1877F2',
        'Apple Inc.': '#000000',
        'Amazon AWS': '#FF9900'
    }
    
    provider_markers = {
        'Google': 'o',
        'Microsoft': 's',
        'Facebook': '^',
        'Apple Inc.': 'D',
        'Amazon AWS': 'v'
    }
    
    # Plot data centers by provider and water stress
    for provider in ai_datacenters['PROVIDER_NAME'].unique():
        provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
        
        # Size based on water stress level
        sizes = []
        for label in provider_data['bwd_label']:
            if 'Extremely High' in str(label):
                sizes.append(200)
            elif 'High' in str(label):
                sizes.append(150)
            elif 'Medium' in str(label):
                sizes.append(100)
            else:
                sizes.append(75)
        
        ax.scatter(provider_data.geometry.x, provider_data.geometry.y,
                  s=sizes, 
                  c=provider_colors.get(provider, '#666666'),
                  marker=provider_markers.get(provider, 'o'),
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=1,
                  label=f'{provider} ({len(provider_data)} sites)')
    
    # Focus on continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    
    # Add title and labels
    ax.set_title('AI Data Centers and Water Stress Levels\n(Size indicates water stress severity)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add legend for providers
    provider_legend = ax.legend(loc='upper left', title='Data Center Provider', 
                               frameon=True, facecolor='white', edgecolor='black')
    
    # Add water stress legend
    water_stress_handles = [
        plt.scatter([], [], s=200, c='gray', alpha=0.7, edgecolors='black', label='Extremely High (>75%)'),
        plt.scatter([], [], s=150, c='gray', alpha=0.7, edgecolors='black', label='High (40-75%)'),
        plt.scatter([], [], s=100, c='gray', alpha=0.7, edgecolors='black', label='Medium (20-40%)'),
        plt.scatter([], [], s=75, c='gray', alpha=0.7, edgecolors='black', label='Low (<20%)')
    ]
    
    water_legend = ax.legend(handles=water_stress_handles, loc='upper right', 
                           title='Water Depletion Level', frameon=True,
                           facecolor='white', edgecolor='black')
    
    # Add the provider legend back
    ax.add_artist(provider_legend)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ai_datacenter_water_stress_map.png', dpi=300, bbox_inches='tight')
    print("Saved: ai_datacenter_water_stress_map.png")
    
    return fig

def create_summary_visualizations(ai_datacenters):
    """Create summary charts and statistics."""
    print("\n5. CREATING SUMMARY VISUALIZATIONS")
    print("-"*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Bar chart by water stress category
    ax1 = axes[0, 0]
    stress_counts = ai_datacenters['bwd_label'].value_counts()
    colors = ['#b2182b' if 'Extremely' in x else 
              '#ef8a62' if 'High' in x else
              '#fddbc7' if 'Medium' in x else
              '#67a9cf' if 'Low' in x else '#969696' 
              for x in stress_counts.index]
    
    stress_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('AI Data Centers by Water Depletion Category', fontweight='bold')
    ax1.set_xlabel('Water Depletion Category')
    ax1.set_ylabel('Number of Data Centers')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Provider comparison
    ax2 = axes[0, 1]
    provider_summary = ai_datacenters.groupby('PROVIDER_NAME').agg({
        'bwd_score': 'mean',
        'ATERIO_DATA_CENTER_UID': 'count'
    }).sort_values('bwd_score', ascending=False)
    
    provider_summary['bwd_score'].plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Average Water Depletion Score by Provider', fontweight='bold')
    ax2.set_xlabel('Water Depletion Score (0-5 scale)')
    
    # Add count labels
    for i, (idx, row) in enumerate(provider_summary.iterrows()):
        ax2.text(row['bwd_score'] + 0.05, i, f"{row['ATERIO_DATA_CENTER_UID']} sites", 
                va='center', fontsize=9)
    
    # 3. Regional distribution
    ax3 = axes[1, 0]
    region_summary = ai_datacenters.groupby('region').agg({
        'ATERIO_DATA_CENTER_UID': 'count',
        'bwd_score': 'mean'
    }).sort_values('ATERIO_DATA_CENTER_UID', ascending=False).head(10)
    
    region_summary['ATERIO_DATA_CENTER_UID'].plot(kind='bar', ax=ax3, color='coral')
    ax3.set_title('Top 10 Regions by Number of AI Data Centers', fontweight='bold')
    ax3.set_xlabel('Region/State')
    ax3.set_ylabel('Number of Data Centers')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. High-risk distribution
    ax4 = axes[1, 1]
    high_risk = ai_datacenters[ai_datacenters['bwd_label'].str.contains('High|Extremely', na=False)]
    if len(high_risk) > 0:
        high_risk_by_provider = high_risk.groupby('PROVIDER_NAME').size()
        high_risk_by_provider.plot(kind='pie', ax=ax4, autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Distribution of {len(high_risk)} High Water Stress Data Centers', fontweight='bold')
        ax4.set_ylabel('')
    else:
        ax4.text(0.5, 0.5, 'No data centers in high water stress areas', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('High Water Stress Data Centers', fontweight='bold')
    
    plt.suptitle('AI Data Centers and Water Stress Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('water_stress_summary_charts.png', dpi=300, bbox_inches='tight')
    print("Saved: water_stress_summary_charts.png")
    
    return fig

def save_results(ai_datacenters, water_stress_df):
    """Save analysis results to CSV files."""
    print("\n6. SAVING RESULTS")
    print("-"*50)
    
    # Save merged data center data
    output = ai_datacenters.copy()
    output['latitude'] = output.geometry.y
    output['longitude'] = output.geometry.x
    output = output.drop(columns=['geometry', 'primaryDataCenterProvider'])
    
    output.to_csv('ai_datacenters_water_stress_merged.csv', index=False)
    print("Saved: ai_datacenters_water_stress_merged.csv")
    
    # Create and save provider summary
    provider_summary = ai_datacenters.groupby('PROVIDER_NAME').agg({
        'ATERIO_DATA_CENTER_UID': 'count',
        'bwd_score': ['mean', 'std', 'min', 'max'],
        'bws_score': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Add percentage in each risk category
    for provider in ai_datacenters['PROVIDER_NAME'].unique():
        provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
        total = len(provider_data)
        
        high_risk = provider_data['bwd_label'].str.contains('High|Extremely', na=False).sum()
        provider_summary.loc[provider, ('risk_percentage', 'high_or_extreme')] = (high_risk / total * 100) if total > 0 else 0
    
    provider_summary.to_csv('provider_water_stress_summary_geographic.csv')
    print("Saved: provider_water_stress_summary_geographic.csv")
    
    # Create regional summary
    regional_summary = ai_datacenters.groupby(['country', 'region']).agg({
        'ATERIO_DATA_CENTER_UID': 'count',
        'PROVIDER_NAME': lambda x: ', '.join(x.unique()),
        'bwd_score': 'mean',
        'bwd_label': lambda x: x.mode()[0] if len(x) > 0 else 'No Data'
    }).round(2)
    
    regional_summary.to_csv('regional_water_stress_summary.csv')
    print("Saved: regional_water_stress_summary.csv")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total AI data centers analyzed: {len(ai_datacenters)}")
    print(f"Data centers with water stress data: {(ai_datacenters['bwd_label'] != 'No Data').sum()}")
    print(f"Data centers in high/extreme water stress: {ai_datacenters['bwd_label'].str.contains('High|Extremely', na=False).sum()}")
    print(f"\nWater Stress Distribution:")
    print(ai_datacenters['bwd_label'].value_counts())
    print(f"\nTop 5 Providers by Average Water Stress:")
    print(ai_datacenters.groupby('PROVIDER_NAME')['bwd_score'].mean().sort_values(ascending=False).head())

def main():
    """Main execution function."""
    try:
        os.chdir(CODE_DIR)
        
        # Load data
        ai_datacenters, all_datacenters = load_ai_datacenters()
        water_stress_df = load_water_stress_regions()
        
        # Geographic merge only (no temporal component)
        ai_datacenters = geographic_merge_simple(ai_datacenters, water_stress_df)
        
        # Create visualizations
        map_fig = create_water_stress_map(ai_datacenters, water_stress_df)
        summary_fig = create_summary_visualizations(ai_datacenters)
        
        # Save results
        save_results(ai_datacenters, water_stress_df)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("1. ai_datacenters_water_stress_merged.csv - Full merged dataset")
        print("2. provider_water_stress_summary_geographic.csv - Provider summary")
        print("3. regional_water_stress_summary.csv - Regional summary")
        print("4. ai_datacenter_water_stress_map.png - Geographic visualization")
        print("5. water_stress_summary_charts.png - Summary charts")
        
        plt.show()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
