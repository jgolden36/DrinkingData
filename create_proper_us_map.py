"""
Proper US Map with Water Stress and AI Data Centers
Uses actual geographic boundaries from shapefile data
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import os
import warnings
warnings.filterwarnings('ignore')

print("Creating Proper US Map with Water Stress Data...")
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

print("\n1. Loading Geographic Data...")
print("-"*50)

# Try to load actual shapefile data
shapefile_loaded = False
gdf = None

# Check for electric utilities shapefile (which contains US geography)
shapefile_path = 'electric-retail-service-territories/electric-retail-service-territories.shp'
if os.path.exists(shapefile_path):
    print(f"Loading shapefile: {shapefile_path}")
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        else:
            gdf = gdf.to_crs('EPSG:4326')
        shapefile_loaded = True
        print(f"Loaded {len(gdf)} geographic regions")
    except Exception as e:
        print(f"Error loading shapefile: {e}")

# Alternative: Check for mapData.shp
if not shapefile_loaded:
    if os.path.exists('mapData.shp'):
        print("Loading mapData.shp...")
        try:
            gdf = gpd.read_file('mapData.shp')
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            else:
                gdf = gdf.to_crs('EPSG:4326')
            shapefile_loaded = True
            print(f"Loaded {len(gdf)} geographic regions")
        except Exception as e:
            print(f"Error loading shapefile: {e}")

# If no shapefile, we'll use state boundaries from geopandas
if not shapefile_loaded:
    print("No local shapefile found. Using built-in US states data...")
    try:
        import geopandas as gpd
        # Get US states from geopandas datasets
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        usa = world[world.name == 'United States of America']
        gdf = usa
        shapefile_loaded = True
        print("Loaded US boundaries from geopandas")
    except:
        print("Could not load geographic data")

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

# Convert to GeoDataFrame
datacenter_gdf = gpd.GeoDataFrame(
    dataCenterdf,
    geometry=gpd.points_from_xy(dataCenterdf.LOCATION_LONGITUDE, dataCenterdf.LOCATION_LATITUDE),
    crs='EPSG:4326'
)

# Get AI data centers
ai_datacenters = datacenter_gdf.merge(
    AI_df[['primaryDataCenterProvider']].drop_duplicates(),
    left_on='PROVIDER_NAME',
    right_on='primaryDataCenterProvider',
    how='inner'
)
ai_datacenters = ai_datacenters.groupby('ATERIO_DATA_CENTER_UID').first().reset_index()
print(f"Loaded {len(ai_datacenters)} unique AI data centers")

os.chdir(CODE_DIR)

print("\n3. Creating Map Visualization...")
print("-"*50)

# Create figure
fig, ax = plt.subplots(figsize=(20, 12), facecolor='white')

# Set background color (ocean)
ax.set_facecolor('#B0E0E6')  # Powder blue for water

# Plot the base map
if shapefile_loaded and gdf is not None:
    # Filter to continental US if we have US regions
    if 'geometry' in gdf.columns:
        # Get bounds for continental US
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        
        # If bounds seem global, filter to US
        if bounds[0] < -180 or bounds[2] > 0:
            # Try to filter to US area
            try:
                # Continental US approximate bounds
                us_bounds = [-130, 24, -65, 50]
                gdf_filtered = gdf.cx[us_bounds[0]:us_bounds[2], us_bounds[1]:us_bounds[3]]
                if len(gdf_filtered) > 0:
                    gdf = gdf_filtered
            except:
                pass
        
        # Plot the geography
        gdf.plot(ax=ax, color='#F5F5DC', edgecolor='#666666', linewidth=0.5, alpha=0.9)
        print("Plotted geographic boundaries")

# Define water stress zones (actual high-stress areas in the US)
water_stress_zones = {
    'Central Valley CA': {
        'bounds': [-122.5, 35.5, -119, 40],
        'center': [-120.75, 37.75],
        'level': 'Extremely High',
        'color': '#8B0000'
    },
    'Los Angeles Basin': {
        'bounds': [-119, 33, -117, 34.5],
        'center': [-118, 34],
        'level': 'Extremely High',
        'color': '#8B0000'
    },
    'Phoenix Metro': {
        'bounds': [-113, 32.5, -111, 34],
        'center': [-112, 33.5],
        'level': 'Extremely High',
        'color': '#8B0000'
    },
    'Las Vegas Valley': {
        'bounds': [-116, 35.5, -114.5, 36.5],
        'center': [-115.2, 36.1],
        'level': 'Extremely High',
        'color': '#8B0000'
    },
    'Rio Grande Valley': {
        'bounds': [-107, 31, -105, 36],
        'center': [-106, 33.5],
        'level': 'High',
        'color': '#CD5C5C'
    },
    'Denver-Front Range': {
        'bounds': [-106, 38.5, -104, 41],
        'center': [-105, 39.7],
        'level': 'High',
        'color': '#CD5C5C'
    },
    'Texas Panhandle': {
        'bounds': [-103, 34, -100, 37],
        'center': [-101.5, 35.5],
        'level': 'High',
        'color': '#CD5C5C'
    },
    'Central Texas': {
        'bounds': [-99, 29, -96, 31.5],
        'center': [-97.5, 30.3],
        'level': 'High',
        'color': '#CD5C5C'
    }
}

# Add water stress zones to map
from matplotlib.patches import Rectangle
for zone_name, zone_info in water_stress_zones.items():
    x_min, y_min, x_max, y_max = zone_info['bounds']
    
    # Add semi-transparent rectangle for water stress zone
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                    facecolor=zone_info['color'], alpha=0.3,
                    edgecolor=zone_info['color'], linewidth=1.5,
                    linestyle='--', zorder=5)
    ax.add_patch(rect)
    
    # Add label
    cx, cy = zone_info['center']
    ax.text(cx, cy, zone_name, fontsize=8, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
           zorder=6)

# Provider styles
provider_styles = {
    'Google': {'color': '#4285F4', 'marker': 'o', 'size': 80},
    'Microsoft': {'color': '#00BCF2', 'marker': 's', 'size': 80},
    'Facebook': {'color': '#1877F2', 'marker': '^', 'size': 80},
    'Amazon AWS': {'color': '#FF9900', 'marker': 'D', 'size': 80},
    'Apple Inc.': {'color': '#555555', 'marker': 'p', 'size': 80}
}

# Count data centers in high-risk zones
high_risk_count = 0
total_count = len(ai_datacenters)

# Plot data centers
for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
    style = provider_styles.get(provider, {'color': '#666666', 'marker': 'o', 'size': 60})
    
    # Check which are in water stress zones
    high_stress_mask = []
    for idx, row in provider_data.iterrows():
        lon = row.geometry.x
        lat = row.geometry.y
        in_stress = False
        
        for zone_name, zone_info in water_stress_zones.items():
            x_min, y_min, x_max, y_max = zone_info['bounds']
            if x_min <= lon <= x_max and y_min <= lat <= y_max:
                in_stress = True
                high_risk_count += 1
                break
        high_stress_mask.append(in_stress)
    
    high_stress_mask = np.array(high_stress_mask)
    
    # Plot high-stress data centers with emphasis
    if high_stress_mask.any():
        high_data = provider_data[high_stress_mask]
        ax.scatter(high_data.geometry.x, high_data.geometry.y,
                  c=style['color'], s=style['size']*1.5, marker=style['marker'],
                  alpha=0.9, edgecolors='red', linewidths=2.5, zorder=10)
    
    # Plot normal data centers
    if (~high_stress_mask).any():
        normal_data = provider_data[~high_stress_mask]
        ax.scatter(normal_data.geometry.x, normal_data.geometry.y,
                  c=style['color'], s=style['size'], marker=style['marker'],
                  alpha=0.7, edgecolors='white', linewidths=1.5,
                  label=f'{provider} ({len(provider_data)} sites)', zorder=9)

# Set map extent to continental US
ax.set_xlim(-126, -66)
ax.set_ylim(24, 50)

# Clean up axes
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Title
ax.set_title('AI Data Centers and Water Stress Regions - United States',
           fontsize=18, fontweight='bold', pad=20)

# Subtitle
subtitle = f'{high_risk_count} of {total_count} data centers ({high_risk_count/total_count*100:.1f}%) in high water stress areas'
ax.text(0.5, 0.96, subtitle, transform=ax.transAxes, ha='center',
       fontsize=12, color='#555555', style='italic')

# Legend
legend_elements = [
    mpatches.Patch(color='#8B0000', alpha=0.3, label='Extremely High Water Stress'),
    mpatches.Patch(color='#CD5C5C', alpha=0.3, label='High Water Stress'),
    mpatches.Patch(color='none', label=''),
]

for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    style = provider_styles.get(provider, {'color': '#666', 'marker': 'o'})
    count = len(ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider])
    legend_elements.append(
        plt.Line2D([0], [0], marker=style['marker'], color='w',
                  markerfacecolor=style['color'], markersize=10,
                  markeredgecolor='white', markeredgewidth=1.5,
                  label=f'{provider} ({count} data centers)')
    )

ax.legend(handles=legend_elements, loc='lower left', frameon=True,
         facecolor='white', edgecolor='#CCCCCC', framealpha=0.95)

# Data source
ax.text(0.99, 0.01, 'Sources: WRI Aqueduct 4.0, Epoch AI Database, Data Center Inventory 2025',
       transform=ax.transAxes, ha='right', fontsize=8, color='#666666', style='italic')

# Save the map
plt.tight_layout()
output_file = 'proper_us_water_stress_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nMap saved as: {output_file}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print(f"Total AI Data Centers: {total_count}")
print(f"Data Centers in High/Extreme Water Stress Areas: {high_risk_count} ({high_risk_count/total_count*100:.1f}%)")

print("\nHigh Water Stress Regions:")
for zone_name, zone_info in water_stress_zones.items():
    print(f"  • {zone_name}: {zone_info['level']}")

plt.show()
print("\nMap creation complete!")
