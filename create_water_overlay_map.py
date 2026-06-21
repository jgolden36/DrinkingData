"""
Simple Water Depletion Map with AI Data Center Overlay
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

print("Creating Water Depletion Overlay Map...")
print("="*60)

# Set directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')

os.chdir(DATA_DIR)

# Helper function
def map_organization_to_provider(organization):
    if pd.isna(organization):
        return None, None
    org_lower = str(organization).lower()
    providers = []
    if any(k in org_lower for k in ['google', 'anthropic', 'deep mind', 'deepmind']):
        providers.append('Google')
    if any(k in org_lower for k in ['microsoft', 'openai']):
        providers.append('Microsoft')
    if any(k in org_lower for k in ['meta', 'facebook']):
        providers.append('Facebook')
    if 'apple' in org_lower:
        providers.append('Apple Inc.')
    if any(k in org_lower for k in ['amazon', 'perplexity']):
        providers.append('Amazon AWS')
    providers = list(dict.fromkeys(providers))
    return (providers[0] if providers else None, providers[1] if len(providers) > 1 else None)

# Load AI models
print("Loading AI models...")
AI_df = pd.read_csv('Epoch Database - Notable Models.csv', thousands=',')
AI_df["Publication date"] = pd.to_datetime(AI_df["Publication date"])
AI_df[['primaryDataCenterProvider', 'secondaryDataCenterProvider']] = AI_df['Organization'].apply(
    lambda x: pd.Series(map_organization_to_provider(x))
)
AI_df = AI_df[(AI_df['Frontier model'] == 'checked') & (AI_df['primaryDataCenterProvider'].notna())]
print(f"Found {len(AI_df)} frontier AI models")

# Load data centers
print("Loading data centers...")
dataCenterdf = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')
dataCenterdf['geometry'] = dataCenterdf.apply(
    lambda row: Point(row['LOCATION_LONGITUDE'], row['LOCATION_LATITUDE']), axis=1
)

# Get AI data centers
ai_datacenters = dataCenterdf.merge(
    AI_df[['primaryDataCenterProvider']].drop_duplicates(),
    left_on='PROVIDER_NAME',
    right_on='primaryDataCenterProvider',
    how='inner'
)

# Remove duplicates
ai_datacenters = ai_datacenters.groupby('ATERIO_DATA_CENTER_UID').first().reset_index()
print(f"Found {len(ai_datacenters)} unique AI data centers")

# Define high water depletion zones
high_water_zones = [
    # California Central Valley & Coast
    {'name': 'California', 'bounds': {'lon': (-125, -114), 'lat': (32, 42)}, 'level': 'Extremely High'},
    # Southwest (Arizona, Nevada)
    {'name': 'Southwest', 'bounds': {'lon': (-117, -109), 'lat': (31, 37)}, 'level': 'Extremely High'},
    # Texas
    {'name': 'Texas', 'bounds': {'lon': (-107, -93), 'lat': (25, 37)}, 'level': 'High'},
    # Colorado Basin
    {'name': 'Colorado Basin', 'bounds': {'lon': (-114, -105), 'lat': (35, 42)}, 'level': 'High'},
]

os.chdir(CODE_DIR)

# Create the map
print("\nCreating map...")
fig, ax = plt.subplots(figsize=(16, 10))

# Draw US outline (simplified)
us_bounds = Rectangle((-125, 24), 60, 25, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(us_bounds)

# Draw high water depletion zones
for zone in high_water_zones:
    lon_min, lon_max = zone['bounds']['lon']
    lat_min, lat_max = zone['bounds']['lat']
    
    color = '#8b0000' if zone['level'] == 'Extremely High' else '#ff4500'
    alpha = 0.3 if zone['level'] == 'Extremely High' else 0.25
    
    rect = Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                     facecolor=color, alpha=alpha, edgecolor='darkred', linewidth=1.5)
    ax.add_patch(rect)
    
    # Add zone label
    ax.text((lon_min+lon_max)/2, (lat_min+lat_max)/2, zone['name'],
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Provider styles
provider_colors = {
    'Google': '#4285F4',
    'Microsoft': '#00A4EF',
    'Facebook': '#1877F2',
    'Apple Inc.': '#555555',
    'Amazon AWS': '#FF9900'
}

provider_markers = {
    'Google': 'o',
    'Microsoft': 's',
    'Facebook': '^',
    'Apple Inc.': 'D',
    'Amazon AWS': 'v'
}

# Count data centers in high-risk zones
high_risk_count = 0
total_count = len(ai_datacenters)

# Plot data centers
for provider in ai_datacenters['PROVIDER_NAME'].unique():
    provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
    
    # Check which are in high zones
    in_high = []
    for _, dc in provider_data.iterrows():
        is_high = False
        for zone in high_water_zones:
            lon_min, lon_max = zone['bounds']['lon']
            lat_min, lat_max = zone['bounds']['lat']
            if lon_min <= dc['LOCATION_LONGITUDE'] <= lon_max and lat_min <= dc['LOCATION_LATITUDE'] <= lat_max:
                is_high = True
                high_risk_count += 1
                break
        in_high.append(is_high)
    
    in_high = np.array(in_high)
    
    # Plot high-risk data centers
    if in_high.any():
        high_data = provider_data[in_high]
        ax.scatter(high_data['LOCATION_LONGITUDE'], high_data['LOCATION_LATITUDE'],
                  s=150, c=provider_colors.get(provider, '#666'),
                  marker=provider_markers.get(provider, 'o'),
                  alpha=0.9, edgecolors='red', linewidth=2.5,
                  label=f'{provider} (High Risk)')
    
    # Plot normal data centers  
    if (~in_high).any():
        normal_data = provider_data[~in_high]
        ax.scatter(normal_data['LOCATION_LONGITUDE'], normal_data['LOCATION_LATITUDE'],
                  s=100, c=provider_colors.get(provider, '#666'),
                  marker=provider_markers.get(provider, 'o'),
                  alpha=0.7, edgecolors='black', linewidth=1,
                  label=f'{provider}')

# Set map bounds
ax.set_xlim(-130, -65)
ax.set_ylim(22, 50)

# Labels and title
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'AI Data Centers and High Water Depletion Zones\n{high_risk_count} of {total_count} data centers in high-risk areas',
           fontsize=16, fontweight='bold', pad=20)

# Create legend
legend_elements = [
    mpatches.Patch(color='#8b0000', alpha=0.3, label='Extremely High Depletion'),
    mpatches.Patch(color='#ff4500', alpha=0.25, label='High Depletion'),
    mpatches.Patch(color='white', label=''),  # Spacer
]

# Add provider markers
for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    n = len(ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider])
    legend_elements.append(
        plt.Line2D([0], [0], marker=provider_markers.get(provider, 'o'), color='w',
                  markerfacecolor=provider_colors.get(provider, '#666'),
                  markersize=10, label=f'{provider} ({n} sites)')
    )

ax.legend(handles=legend_elements, loc='upper left', frameon=True,
         facecolor='white', edgecolor='black')

# Add grid
ax.grid(True, alpha=0.3, linestyle=':')

# Save
plt.tight_layout()
output_file = 'water_depletion_overlay_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nMap saved as: {output_file}")

# Create summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total AI Data Centers: {total_count}")
print(f"Data Centers in High Water Risk Zones: {high_risk_count}")
print(f"Percentage at Risk: {high_risk_count/total_count*100:.1f}%")

# Count by provider in high-risk zones
print("\nHigh-Risk Data Centers by Provider:")
for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
    count = 0
    for _, dc in provider_data.iterrows():
        for zone in high_water_zones:
            lon_min, lon_max = zone['bounds']['lon']
            lat_min, lat_max = zone['bounds']['lat']
            if lon_min <= dc['LOCATION_LONGITUDE'] <= lon_max and lat_min <= dc['LOCATION_LATITUDE'] <= lat_max:
                count += 1
                break
    print(f"  {provider}: {count} data centers")

plt.show()
print("\nMap creation complete!")
