"""
Water Stress and AI Data Center Analysis using GeoPandas
This script merges AI data center locations with water stress data from WRI Aqueduct 4.0
Creates visualizations overlaying data centers on water stress regions.
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import warnings
warnings.filterwarnings('ignore')

print("Starting Water Stress Analysis...")

# Set working directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')
WATER_DATA_DIR = os.path.join(DATA_DIR, 'Water Data', 'Water Resources Institute', 
                             'aqueduct-4-0-water-risk-data', 'Aqueduct40_waterrisk_download_Y2023M07D05')

def map_organization_to_provider(organization):
    """Maps organization names to data center providers."""
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

# Load and filter AI data centers
print("\n1. Loading AI and Data Center data...")
os.chdir(DATA_DIR)

# Load AI models
AI_df = pd.read_csv('Epoch Database - Notable Models.csv', thousands=',')
AI_df["Publication date"] = pd.to_datetime(AI_df["Publication date"])
AI_df[['primaryDataCenterProvider', 'secondaryDataCenterProvider']] = AI_df['Organization'].apply(
    lambda x: pd.Series(map_organization_to_provider(x))
)
AI_df = AI_df[(AI_df['Frontier model'] == 'checked') & (AI_df['primaryDataCenterProvider'].notna())]
print(f"Found {len(AI_df)} frontier AI models with identified providers")

# Load data centers
dataCenterdf = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')
dataCenterdf['geometry'] = dataCenterdf.apply(
    lambda row: Point(row['LOCATION_LONGITUDE'], row['LOCATION_LATITUDE']), axis=1
)
dataCenters_gdf = gpd.GeoDataFrame(dataCenterdf, geometry='geometry', crs='EPSG:4326')
print(f"Loaded {len(dataCenters_gdf)} total data centers")

# Filter for AI-related data centers
ai_datacenters = dataCenters_gdf.merge(
    AI_df[['primaryDataCenterProvider', 'Model', 'Parameters', 'Publication date', 'Organization']],
    left_on='PROVIDER_NAME',
    right_on='primaryDataCenterProvider',
    how='inner'
)
print(f"Filtered to {len(ai_datacenters)} AI-related data centers")
print(f"Providers: {ai_datacenters['PROVIDER_NAME'].unique()}")

# Load water stress data from CSV
print("\n2. Loading Water Stress data...")
csv_path = os.path.join(WATER_DATA_DIR, 'CVS', 'Aqueduct40_baseline_annual_y2023m07d05.csv')

# Read sample to understand structure
print("Reading water stress data sample...")
water_sample = pd.read_csv(csv_path, nrows=1000)
print(f"Water data columns (sample): {water_sample.columns.tolist()[:20]}")
print(f"BWD Label categories: {water_sample['bwd_label'].unique()}")

# For full analysis, we'll aggregate by region
print("\n3. Processing water stress by region...")
# Read data in chunks and aggregate
chunk_size = 50000
water_summary = {}

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
    if i >= 5:  # Process first 250k rows only for speed
        break
    
    # Group by country and state/region
    for _, row in chunk.groupby(['name_0', 'name_1']).agg({
        'bwd_score': 'mean',
        'bwd_label': lambda x: x.mode()[0] if len(x) > 0 else 'No Data',
        'bws_score': 'mean',
        'bws_label': lambda x: x.mode()[0] if len(x) > 0 else 'No Data'
    }).iterrows():
        key = row.name
        if key not in water_summary:
            water_summary[key] = {
                'bwd_score': [],
                'bws_score': [],
                'bwd_label': [],
                'bws_label': []
            }
        water_summary[key]['bwd_score'].append(row['bwd_score'])
        water_summary[key]['bws_score'].append(row['bws_score'])
        water_summary[key]['bwd_label'].append(row['bwd_label'])
        water_summary[key]['bws_label'].append(row['bws_label'])

# Convert to DataFrame
water_regions = []
for (country, region), values in water_summary.items():
    water_regions.append({
        'country': country,
        'region': region,
        'bwd_score': np.mean([v for v in values['bwd_score'] if pd.notna(v)]),
        'bws_score': np.mean([v for v in values['bws_score'] if pd.notna(v)]),
        'bwd_label': max(set(values['bwd_label']), key=values['bwd_label'].count),
        'bws_label': max(set(values['bws_label']), key=values['bws_label'].count)
    })

water_regions_df = pd.DataFrame(water_regions)
print(f"Processed {len(water_regions_df)} unique regions")
print(f"Countries: {water_regions_df['country'].nunique()}")

# Focus on US regions
us_water = water_regions_df[water_regions_df['country'] == 'United States of America'].copy()
print(f"\nUS regions with water data: {len(us_water)}")
print(us_water[['region', 'bwd_label', 'bws_label']].head(10))

# Assign water stress to data centers (simplified - based on country)
print("\n4. Assigning water stress to data centers...")
ai_datacenters['country'] = 'Unknown'
ai_datacenters['bwd_score'] = np.nan
ai_datacenters['bwd_label'] = 'No Data'
ai_datacenters['bws_score'] = np.nan  
ai_datacenters['bws_label'] = 'No Data'

# Simple geographic assignment for US data centers
us_mask = (ai_datacenters.geometry.x > -130) & (ai_datacenters.geometry.x < -60) & \
          (ai_datacenters.geometry.y > 24) & (ai_datacenters.geometry.y < 50)

if len(us_water) > 0:
    # Assign average US water stress
    avg_bwd = us_water['bwd_score'].mean()
    avg_bws = us_water['bws_score'].mean()
    mode_bwd_label = us_water['bwd_label'].mode()[0] if len(us_water['bwd_label'].mode()) > 0 else 'No Data'
    mode_bws_label = us_water['bws_label'].mode()[0] if len(us_water['bws_label'].mode()) > 0 else 'No Data'
    
    ai_datacenters.loc[us_mask, 'country'] = 'United States'
    ai_datacenters.loc[us_mask, 'bwd_score'] = avg_bwd
    ai_datacenters.loc[us_mask, 'bws_score'] = avg_bws
    ai_datacenters.loc[us_mask, 'bwd_label'] = mode_bwd_label
    ai_datacenters.loc[us_mask, 'bws_label'] = mode_bws_label

print(f"Assigned water stress to {us_mask.sum()} US data centers")

# Create visualization
print("\n5. Creating visualizations...")
os.chdir(CODE_DIR)

# Define color scheme for water stress
water_colors = {
    'Low (<10%)': '#2166ac',
    'Low - Medium (10-20%)': '#67a9cf', 
    'Medium - High (20-40%)': '#fddbc7',
    'High (40-80%)': '#ef8a62',
    'Extremely High (>80%)': '#b2182b',
    'Arid and Low Water Use': '#969696',
    'No Data': '#f7f7f7'
}

# Create main figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Map of US data centers colored by water stress
ax1 = axes[0, 0]
us_ai_data = ai_datacenters[us_mask].copy()

if len(us_ai_data) > 0:
    # Plot by provider with different markers
    markers = {'Google': 'o', 'Microsoft': 's', 'Facebook': '^', 'Apple Inc.': 'D', 'Amazon AWS': 'v'}
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, provider in enumerate(us_ai_data['PROVIDER_NAME'].unique()):
        provider_data = us_ai_data[us_ai_data['PROVIDER_NAME'] == provider]
        ax1.scatter(provider_data.geometry.x, provider_data.geometry.y,
                   s=100, alpha=0.7, label=provider,
                   marker=markers.get(provider, 'o'),
                   c=colors_list[i % len(colors_list)],
                   edgecolors='black', linewidth=1)
    
    ax1.set_xlim(-125, -65)
    ax1.set_ylim(25, 50)
    ax1.set_title('AI Data Centers in United States', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(title='Provider', loc='lower left')
    ax1.grid(True, alpha=0.3)

# 2. Bar chart of data centers by water stress category
ax2 = axes[0, 1]
stress_counts = ai_datacenters['bwd_label'].value_counts()
stress_counts.plot(kind='bar', ax=ax2, color=[water_colors.get(x, '#cccccc') for x in stress_counts.index])
ax2.set_title('AI Data Centers by Water Depletion Category', fontsize=14, fontweight='bold')
ax2.set_xlabel('Water Depletion Category')
ax2.set_ylabel('Number of Data Centers')
ax2.tick_params(axis='x', rotation=45)

# 3. Provider comparison
ax3 = axes[1, 0]
provider_water = ai_datacenters.groupby('PROVIDER_NAME').agg({
    'bwd_score': 'mean',
    'Model': 'count'
}).sort_values('bwd_score')

provider_water['bwd_score'].plot(kind='barh', ax=ax3, color='steelblue')
ax3.set_title('Average Water Depletion Score by AI Provider', fontsize=14, fontweight='bold')
ax3.set_xlabel('Water Depletion Score')
ax3.set_ylabel('Provider')

# Add count annotations
for i, (idx, row) in enumerate(provider_water.iterrows()):
    ax3.text(row['bwd_score'] + 0.5, i, f"({row['Model']} models)", 
            va='center', fontsize=9)

# 4. Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
WATER STRESS ANALYSIS SUMMARY

Total AI Data Centers Analyzed: {len(ai_datacenters)}
Data Centers with Water Data: {(ai_datacenters['bwd_label'] != 'No Data').sum()}

Water Depletion Categories:
{stress_counts.to_string()}

Top Water-Stressed Providers:
{provider_water['bwd_score'].round(2).to_string()}

Geographic Coverage:
- US Data Centers: {us_mask.sum()}
- Other Locations: {(~us_mask).sum()}
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
        fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.suptitle('AI Data Centers and Water Stress Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('water_stress_analysis.png', dpi=300, bbox_inches='tight')
print("Saved water_stress_analysis.png")

# Save data
output_df = ai_datacenters.copy()
output_df['latitude'] = output_df.geometry.y
output_df['longitude'] = output_df.geometry.x
output_df = output_df.drop(columns=['geometry'])
output_df.to_csv('ai_datacenters_water_stress.csv', index=False)
print("Saved ai_datacenters_water_stress.csv")

# Create provider summary table
provider_summary = ai_datacenters.groupby('PROVIDER_NAME').agg({
    'Model': 'nunique',
    'bwd_score': ['mean', 'std'],
    'bws_score': ['mean', 'std'],
    'bwd_label': lambda x: x.mode()[0] if len(x) > 0 else 'No Data'
}).round(2)
provider_summary.to_csv('provider_water_stress_summary.csv')
print("Saved provider_water_stress_summary.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

plt.show()
