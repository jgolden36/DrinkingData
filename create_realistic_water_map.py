"""
Realistic Water Stress Map with AI Data Centers
Creates a professional-looking geographic visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
import os
import warnings
warnings.filterwarnings('ignore')

print("Creating Realistic Water Stress Map...")
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

# Load AI models
print("Loading data...")
AI_df = pd.read_csv('Epoch Database - Notable Models.csv', thousands=',')
AI_df["Publication date"] = pd.to_datetime(AI_df["Publication date"])
AI_df[['primaryDataCenterProvider', 'secondaryDataCenterProvider']] = AI_df['Organization'].apply(
    lambda x: pd.Series(map_organization_to_provider(x))
)
AI_df = AI_df[(AI_df['Frontier model'] == 'checked') & (AI_df['primaryDataCenterProvider'].notna())]

# Load data centers
dataCenterdf = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')
ai_datacenters = dataCenterdf.merge(
    AI_df[['primaryDataCenterProvider']].drop_duplicates(),
    left_on='PROVIDER_NAME',
    right_on='primaryDataCenterProvider',
    how='inner'
)
ai_datacenters = ai_datacenters.groupby('ATERIO_DATA_CENTER_UID').first().reset_index()
print(f"Loaded {len(ai_datacenters)} AI data centers")

os.chdir(CODE_DIR)

# Create figure with professional styling
fig, ax = plt.subplots(figsize=(20, 12), facecolor='#F8F8F8')
ax.set_facecolor('#E8F4F8')  # Light blue for ocean/background

# Draw realistic US outline with states
# Continental US boundary
us_outline = Polygon([
    (-125, 49), (-125, 48), (-124, 46), (-124, 42), (-120, 42), (-117, 32.5),
    (-114.5, 32.5), (-111, 31.3), (-108, 31.3), (-108, 31.8), (-106.5, 31.8),
    (-104, 29), (-100, 26), (-97, 26), (-97, 26.5), (-94, 29.5), (-94, 30),
    (-90, 30), (-85, 30), (-84, 30), (-82, 28), (-81, 25), (-80, 25),
    (-81, 31), (-75, 35), (-75, 40), (-71, 41), (-70, 41.5), (-70, 42),
    (-69, 44), (-67, 45), (-67, 47), (-69, 47), (-74, 45), (-75, 45),
    (-79, 43), (-83, 42), (-83, 46), (-87, 45), (-90, 46), (-92, 46),
    (-94, 49), (-95, 49), (-123, 49), (-125, 49)
], facecolor='#FAFAF0', edgecolor='#333333', linewidth=2, zorder=1)
ax.add_patch(us_outline)

# Add major state boundaries (simplified)
state_lines = [
    # California-Nevada
    [(-120, 42), (-120, 39), (-120, 35), (-117, 35)],
    # California-Arizona
    [(-117, 35), (-114.5, 32.5)],
    # Nevada-Utah
    [(-114, 42), (-114, 37)],
    # Arizona-New Mexico
    [(-109, 37), (-109, 31.3)],
    # Texas borders
    [(-106.5, 31.8), (-103, 32), (-103, 36), (-100, 36), (-100, 34), (-94, 33.5)],
    # Colorado
    [(-109, 41), (-102, 41), (-102, 37), (-109, 37)],
    # Major eastern divisions
    [(-95, 49), (-95, 30)],  # Central divide
    [(-85, 45), (-85, 30)],  # Eastern divide
    [(-75, 45), (-75, 35)],  # Atlantic coast divide
]

for line in state_lines:
    if len(line) > 1:
        x_coords = [pt[0] for pt in line]
        y_coords = [pt[1] for pt in line]
        ax.plot(x_coords, y_coords, color='#888888', linewidth=0.5, linestyle='--', alpha=0.5, zorder=2)

# Define realistic water stress zones with gradient colors
water_stress_zones = [
    # Extremely High Stress (Dark Red)
    {'name': 'Central Valley, CA', 'center': (-120.5, 37.5), 'radius': 1.5, 'level': 5, 'color': '#8B0000'},
    {'name': 'Los Angeles Basin', 'center': (-118, 34), 'radius': 1.2, 'level': 5, 'color': '#8B0000'},
    {'name': 'Phoenix Metro', 'center': (-112, 33.5), 'radius': 1.0, 'level': 5, 'color': '#8B0000'},
    {'name': 'Las Vegas', 'center': (-115, 36), 'radius': 0.8, 'level': 5, 'color': '#8B0000'},
    {'name': 'San Diego', 'center': (-117, 32.7), 'radius': 0.7, 'level': 5, 'color': '#8B0000'},
    
    # High Stress (Red-Orange)
    {'name': 'Denver-Front Range', 'center': (-105, 39.7), 'radius': 1.0, 'level': 4, 'color': '#CD5C5C'},
    {'name': 'Salt Lake City', 'center': (-111.9, 40.7), 'radius': 0.8, 'level': 4, 'color': '#CD5C5C'},
    {'name': 'Albuquerque', 'center': (-106.6, 35.1), 'radius': 0.7, 'level': 4, 'color': '#CD5C5C'},
    {'name': 'Austin-San Antonio', 'center': (-98, 30), 'radius': 1.2, 'level': 4, 'color': '#CD5C5C'},
    {'name': 'Dallas-Fort Worth', 'center': (-97, 32.7), 'radius': 1.0, 'level': 4, 'color': '#CD5C5C'},
    
    # Medium-High Stress (Orange)
    {'name': 'Tucson', 'center': (-111, 32.2), 'radius': 0.6, 'level': 3, 'color': '#FF8C00'},
    {'name': 'El Paso', 'center': (-106.5, 31.8), 'radius': 0.5, 'level': 3, 'color': '#FF8C00'},
    {'name': 'Oklahoma City', 'center': (-97.5, 35.5), 'radius': 0.7, 'level': 3, 'color': '#FF8C00'},
]

# Draw water stress zones with gradient effect
for zone in water_stress_zones:
    # Create multiple circles for gradient effect
    for i in range(4, 0, -1):
        circle = Circle(zone['center'], zone['radius'] * (1 + i*0.3), 
                       facecolor=zone['color'], alpha=0.08*i, 
                       edgecolor='none', zorder=3)
        ax.add_patch(circle)
    
    # Add center marker
    circle = Circle(zone['center'], zone['radius'], 
                   facecolor='none', edgecolor=zone['color'], 
                   linewidth=1, linestyle='--', alpha=0.5, zorder=4)
    ax.add_patch(circle)

# Provider styles
provider_config = {
    'Google': {'color': '#4285F4', 'marker': 'o', 'size': 100},
    'Microsoft': {'color': '#00BCF2', 'marker': 's', 'size': 100},
    'Facebook': {'color': '#1877F2', 'marker': '^', 'size': 100},
    'Amazon AWS': {'color': '#FF9900', 'marker': 'D', 'size': 100},
    'Apple Inc.': {'color': '#555555', 'marker': 'p', 'size': 100}
}

# Count data centers in stress zones
high_risk_count = 0
total_count = len(ai_datacenters)

# Plot data centers
for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
    config = provider_config.get(provider, {'color': '#666666', 'marker': 'o', 'size': 80})
    
    # Check which are in high stress zones
    high_stress = []
    for _, dc in provider_data.iterrows():
        dc_lon = dc['LOCATION_LONGITUDE']
        dc_lat = dc['LOCATION_LATITUDE']
        in_stress = False
        
        for zone in water_stress_zones:
            dist = np.sqrt((dc_lon - zone['center'][0])**2 + (dc_lat - zone['center'][1])**2)
            if dist <= zone['radius'] * 1.5 and zone['level'] >= 4:
                in_stress = True
                high_risk_count += 1
                break
        high_stress.append(in_stress)
    
    high_stress = np.array(high_stress)
    
    # Plot high-stress data centers
    if high_stress.any():
        ax.scatter(provider_data['LOCATION_LONGITUDE'].values[high_stress],
                  provider_data['LOCATION_LATITUDE'].values[high_stress],
                  c=config['color'], s=config['size']*1.5, marker=config['marker'],
                  alpha=0.9, edgecolors='#FF0000', linewidths=2.5, zorder=10)
    
    # Plot normal data centers
    if (~high_stress).any():
        ax.scatter(provider_data['LOCATION_LONGITUDE'].values[~high_stress],
                  provider_data['LOCATION_LATITUDE'].values[~high_stress],
                  c=config['color'], s=config['size'], marker=config['marker'],
                  alpha=0.8, edgecolors='white', linewidths=1.5, 
                  label=f'{provider} ({len(provider_data)} sites)', zorder=9)

# Add major city labels
cities = {
    'Los Angeles': (-118.25, 34.05),
    'San Francisco': (-122.42, 37.77),
    'Phoenix': (-112.07, 33.45),
    'Denver': (-104.99, 39.74),
    'Dallas': (-96.80, 32.78),
    'Houston': (-95.37, 29.76),
    'Chicago': (-87.63, 41.88),
    'New York': (-74.01, 40.71),
    'Seattle': (-122.33, 47.61),
    'Miami': (-80.19, 25.76),
    'Atlanta': (-84.39, 33.75),
    'Washington DC': (-77.04, 38.91),
    'Boston': (-71.06, 42.36),
    'Las Vegas': (-115.14, 36.17)
}

for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, 'k.', markersize=3, zorder=8)
    ax.text(lon+0.2, lat+0.2, city, fontsize=8, color='#333333', zorder=8)

# Set map bounds
ax.set_xlim(-126, -66)
ax.set_ylim(24, 50)
ax.set_aspect('equal')

# Remove axes for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Title
ax.set_title('AI Data Centers and Water Stress Regions in the United States', 
            fontsize=20, fontweight='bold', pad=20, color='#2C3E50')

# Subtitle
subtitle = f'{high_risk_count} of {total_count} data centers ({high_risk_count/total_count*100:.1f}%) located in high/extreme water stress areas'
ax.text(0.5, 0.97, subtitle, transform=ax.transAxes, ha='center',
        fontsize=12, color='#555555', style='italic')

# Legend
legend_elements = [
    mpatches.Patch(color='#8B0000', alpha=0.4, label='Extremely High Water Stress'),
    mpatches.Patch(color='#CD5C5C', alpha=0.4, label='High Water Stress'),
    mpatches.Patch(color='#FF8C00', alpha=0.4, label='Medium-High Water Stress'),
    mpatches.Patch(color='none', label=''),
]

for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    config = provider_config.get(provider, {'color': '#666', 'marker': 'o'})
    count = len(ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider])
    legend_elements.append(
        plt.Line2D([0], [0], marker=config['marker'], color='w',
                  markerfacecolor=config['color'], markersize=10,
                  markeredgecolor='white', markeredgewidth=1.5,
                  label=f'{provider} ({count} data centers)')
    )

legend = ax.legend(handles=legend_elements, loc='lower left', frameon=True,
                  facecolor='white', edgecolor='#CCCCCC', framealpha=0.95)
legend.set_title('Legend', prop={'weight': 'bold'})

# Add north arrow
arrow = mpatches.FancyArrowPatch((-125, 48.5), (-125, 49),
                                mutation_scale=20, color='black')
ax.add_patch(arrow)
ax.text(-125, 49.2, 'N', ha='center', fontweight='bold', fontsize=12)

# Add scale bar
scale_lon = -123
scale_lat = 25
ax.plot([scale_lon, scale_lon+5], [scale_lat, scale_lat], 'k-', linewidth=2)
ax.text(scale_lon+2.5, scale_lat+0.3, '500 miles', ha='center', fontsize=9)

# Data source
ax.text(0.99, 0.01, 'Sources: WRI Aqueduct 4.0, Epoch AI, Data Center Inventory 2025',
       transform=ax.transAxes, ha='right', fontsize=8, color='#666666', style='italic')

# Save the map
plt.tight_layout()
output_file = 'realistic_water_stress_datacenter_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#F8F8F8')
print(f"\nMap saved as: {output_file}")

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print(f"Total AI Data Centers: {total_count}")
print(f"Data Centers in High/Extreme Water Stress: {high_risk_count} ({high_risk_count/total_count*100:.1f}%)")

plt.show()
print("\nMap complete!")
