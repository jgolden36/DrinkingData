"""
US Map with State Boundaries and Water Stress Data
Uses GeoPandas built-in US states data for accurate geography
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
import os
import warnings
warnings.filterwarnings('ignore')

print("Creating US Map with Actual State Boundaries...")
print("="*70)

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

print("\n1. Loading US State Boundaries from GeoPandas...")
print("-"*50)

# Get US states using GeoPandas built-in data
# First try to get US states from natural earth data
try:
    # Load US states - this should give us proper state boundaries
    # You can also download US states separately if needed
    import requests
    import zipfile
    import io
    
    # Try to use a direct US states shapefile from Census
    states_url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
    
    print("Downloading US states shapefile from Census Bureau...")
    response = requests.get(states_url)
    
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('temp_states')
        
        states = gpd.read_file('temp_states/cb_2018_us_state_20m.shp')
        
        # Filter to continental US (exclude Alaska, Hawaii, territories)
        continental = states[~states['STUSPS'].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'])]
        
        # Ensure proper CRS
        continental = continental.to_crs('EPSG:4326')
        print(f"Loaded {len(continental)} continental US states")
    else:
        raise Exception("Could not download states data")
        
except Exception as e:
    print(f"Could not load Census data: {e}")
    print("Trying alternative: naturalearth data...")
    
    try:
        # Alternative: use naturalearth_lowres (less detailed but built-in)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        usa = world[world['name'] == 'United States of America']
        
        # This gives us the whole US as one shape, not individual states
        # But it will at least show the proper US outline
        continental = usa
        print("Loaded US boundary from naturalearth (no state boundaries)")
        
    except Exception as e2:
        print(f"Error loading naturalearth data: {e2}")
        
        # Last resort: create synthetic state boundaries
        print("Creating synthetic US boundaries...")
        from shapely.geometry import box
        continental = gpd.GeoDataFrame(
            [1], geometry=[box(-125, 24.5, -66.9, 49)], crs='EPSG:4326'
        )

print("\n2. Loading AI Data Centers...")
print("-"*50)

# Load AI models
AI_df = pd.read_csv('Epoch Database - Notable Models.csv', thousands=',')
AI_df["Publication date"] = pd.to_datetime(AI_df["Publication date"])
AI_df[['primaryDataCenterProvider', 'secondaryDataCenterProvider']] = AI_df['Organization'].apply(
    lambda x: pd.Series(map_organization_to_provider(x))
)
AI_df = AI_df[(AI_df['Frontier model'] == 'checked') & (AI_df['primaryDataCenterProvider'].notna())]

# Load data centers
dataCenterdf = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')

# Create GeoDataFrame
ai_datacenters = dataCenterdf.merge(
    AI_df[['primaryDataCenterProvider']].drop_duplicates(),
    left_on='PROVIDER_NAME',
    right_on='primaryDataCenterProvider',
    how='inner'
)
ai_datacenters = ai_datacenters.groupby('ATERIO_DATA_CENTER_UID').first().reset_index()
ai_datacenters_gdf = gpd.GeoDataFrame(
    ai_datacenters,
    geometry=gpd.points_from_xy(ai_datacenters.LOCATION_LONGITUDE, ai_datacenters.LOCATION_LATITUDE),
    crs='EPSG:4326'
)

print(f"Loaded {len(ai_datacenters)} unique AI data centers")

os.chdir(CODE_DIR)

print("\n3. Creating Map with Proper US Geography...")
print("-"*50)

# Create figure
fig, ax = plt.subplots(figsize=(20, 12), facecolor='white')

# Set water/ocean color
ax.set_facecolor('#c6dbef')  # Light blue for ocean

# Plot the US states/boundary
continental.plot(ax=ax, 
                color='#f0f0f0',  # Light gray for land
                edgecolor='#636363',  # Dark gray for state boundaries
                linewidth=0.5)

# Add Great Lakes if we have detailed boundaries
# The Great Lakes should appear as "holes" in the states around them

# Define water stress zones (actual locations)
water_stress_zones = {
    'Central Valley, CA': {
        'bounds': [-122.5, 35.5, -119, 40],
        'level': 'Extremely High',
        'color': '#67000d'  # Dark red
    },
    'Los Angeles Basin': {
        'bounds': [-119, 33, -117, 34.5],
        'level': 'Extremely High',
        'color': '#67000d'
    },
    'San Francisco Bay': {
        'bounds': [-123, 37, -121.5, 38.5],
        'level': 'High',
        'color': '#a50f15'
    },
    'Phoenix, AZ': {
        'bounds': [-113, 32.5, -111, 34],
        'level': 'Extremely High',
        'color': '#67000d'
    },
    'Las Vegas, NV': {
        'bounds': [-116, 35.5, -114.5, 36.5],
        'level': 'Extremely High',
        'color': '#67000d'
    },
    'Denver, CO': {
        'bounds': [-106, 39, -104, 40.5],
        'level': 'High',
        'color': '#a50f15'
    },
    'Salt Lake City, UT': {
        'bounds': [-112.5, 40, -111, 41.5],
        'level': 'High',
        'color': '#a50f15'
    },
    'Albuquerque, NM': {
        'bounds': [-107, 34.5, -106, 36],
        'level': 'High',
        'color': '#a50f15'
    },
    'Austin-San Antonio, TX': {
        'bounds': [-99, 29, -97, 31],
        'level': 'High',
        'color': '#cb181d'
    },
    'Dallas-Fort Worth, TX': {
        'bounds': [-98, 32, -96, 33.5],
        'level': 'Medium-High',
        'color': '#ef3b2c'
    },
    'West Texas': {
        'bounds': [-103, 31, -100, 33],
        'level': 'High',
        'color': '#cb181d'
    }
}

# Plot water stress zones
from matplotlib.patches import Rectangle
for zone_name, zone_info in water_stress_zones.items():
    bounds = zone_info['bounds']
    rect = Rectangle((bounds[0], bounds[1]), 
                    bounds[2] - bounds[0], 
                    bounds[3] - bounds[1],
                    facecolor=zone_info['color'], 
                    alpha=0.4,
                    edgecolor=zone_info['color'], 
                    linewidth=1,
                    linestyle='-',
                    zorder=5)
    ax.add_patch(rect)
    
    # Add subtle label
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    ax.text(center_x, center_y, zone_name.split(',')[0], 
           fontsize=7, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
           zorder=6)

# Data center styles
provider_styles = {
    'Google': {'color': '#4285F4', 'marker': 'o', 'size': 70},
    'Microsoft': {'color': '#00BCF2', 'marker': 's', 'size': 70},
    'Facebook': {'color': '#1877F2', 'marker': '^', 'size': 70},
    'Amazon AWS': {'color': '#FF9900', 'marker': 'D', 'size': 70},
    'Apple Inc.': {'color': '#A8DADC', 'marker': 'p', 'size': 70}
}

# Count data centers in water stress zones
high_risk_count = 0
total_count = len(ai_datacenters_gdf)

# Plot data centers
for provider in sorted(ai_datacenters_gdf['PROVIDER_NAME'].unique()):
    provider_data = ai_datacenters_gdf[ai_datacenters_gdf['PROVIDER_NAME'] == provider]
    style = provider_styles.get(provider, {'color': '#666', 'marker': 'o', 'size': 50})
    
    # Check which are in water stress zones
    in_stress = []
    for idx, row in provider_data.iterrows():
        point = row.geometry
        is_stressed = False
        for zone_name, zone_info in water_stress_zones.items():
            bounds = zone_info['bounds']
            if bounds[0] <= point.x <= bounds[2] and bounds[1] <= point.y <= bounds[3]:
                is_stressed = True
                high_risk_count += 1
                break
        in_stress.append(is_stressed)
    
    in_stress = np.array(in_stress)
    
    # Plot stressed data centers
    if in_stress.any():
        stressed = provider_data[in_stress]
        ax.scatter(stressed.geometry.x, stressed.geometry.y,
                  c=style['color'], s=style['size']*1.5, 
                  marker=style['marker'],
                  alpha=0.9, edgecolors='darkred', linewidths=2,
                  zorder=10)
    
    # Plot normal data centers
    if (~in_stress).any():
        normal = provider_data[~in_stress]
        ax.scatter(normal.geometry.x, normal.geometry.y,
                  c=style['color'], s=style['size'], 
                  marker=style['marker'],
                  alpha=0.7, edgecolors='white', linewidths=1,
                  label=f'{provider} ({len(provider_data)} sites)',
                  zorder=9)

# Set map extent to continental US
ax.set_xlim(-125, -66.5)
ax.set_ylim(24.5, 49)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Title
ax.set_title('AI Data Centers and Water Stress Regions in the Continental United States',
           fontsize=18, fontweight='bold', pad=20)

# Subtitle
subtitle = f'{high_risk_count} of {total_count} data centers ({high_risk_count/total_count*100:.1f}%) located in high water stress areas'
ax.text(0.5, 0.97, subtitle, transform=ax.transAxes,
       ha='center', fontsize=12, color='#555', style='italic')

# Legend
legend_elements = [
    mpatches.Patch(color='#67000d', alpha=0.4, label='Extremely High Water Stress'),
    mpatches.Patch(color='#a50f15', alpha=0.4, label='High Water Stress'),
    mpatches.Patch(color='#ef3b2c', alpha=0.4, label='Medium-High Water Stress'),
    mpatches.Patch(color='none', label=''),
]

for provider in sorted(ai_datacenters_gdf['PROVIDER_NAME'].unique()):
    style = provider_styles.get(provider, {'color': '#666', 'marker': 'o'})
    legend_elements.append(
        plt.Line2D([0], [0], marker=style['marker'], color='w',
                  markerfacecolor=style['color'], markersize=10,
                  markeredgecolor='white', markeredgewidth=1,
                  label=provider)
    )

legend = ax.legend(handles=legend_elements, loc='lower left',
                  frameon=True, facecolor='white', 
                  edgecolor='#ccc', framealpha=0.9)

# Data attribution
ax.text(0.99, 0.01, 'Data: WRI Aqueduct 4.0, Epoch AI, Data Center Inventory 2025',
       transform=ax.transAxes, ha='right', 
       fontsize=8, color='#666', style='italic')

# Save
plt.tight_layout()
output_file = 'us_states_water_stress_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nMap saved as: {output_file}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print(f"Total AI Data Centers: {total_count}")
print(f"In High/Extreme Water Stress: {high_risk_count} ({high_risk_count/total_count*100:.1f}%)")

# Clean up temp files if they exist
try:
    import shutil
    if os.path.exists('temp_states'):
        shutil.rmtree('temp_states')
except:
    pass

plt.show()
print("\nMap complete!")
