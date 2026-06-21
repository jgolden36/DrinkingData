"""
Enhanced Water Depletion Map with US and State Boundaries
Shows AI Data Centers overlaid on high water depletion zones with geographic boundaries
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon as MplPolygon
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

print("Creating Enhanced Water Depletion Map with Boundaries...")
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

# Define state boundaries (simplified coordinates for major states)
state_boundaries = {
    'California': [(-124.5, 42), (-120, 42), (-117, 35), (-114.5, 32.5), (-117, 32.5), (-120.5, 34), (-124.5, 40)],
    'Nevada': [(-120, 42), (-114, 42), (-114, 36), (-115, 35), (-117, 35), (-120, 39)],
    'Arizona': [(-114.8, 37), (-109, 37), (-109, 31.3), (-114.8, 32.5)],
    'New Mexico': [(-109, 37), (-103, 37), (-103, 32), (-106.5, 31.8), (-108.2, 31.8), (-109, 32)],
    'Texas': [(-106.5, 31.8), (-103, 32), (-103, 36), (-100, 36), (-100, 34), (-97, 34), 
             (-94, 33.5), (-94, 30), (-97, 26), (-103, 29), (-106.5, 31.8)],
    'Oregon': [(-124.5, 46), (-116.5, 46), (-116.5, 42), (-120, 42), (-124.5, 42)],
    'Washington': [(-124.7, 49), (-117, 49), (-116.5, 46), (-124.5, 46)],
    'Idaho': [(-117, 49), (-111, 49), (-111, 42), (-116.5, 42), (-116.5, 46)],
    'Utah': [(-114, 42), (-109, 42), (-109, 37), (-114, 37)],
    'Colorado': [(-109, 41), (-102, 41), (-102, 37), (-109, 37)],
    'Montana': [(-116, 49), (-104, 49), (-104, 45), (-111, 45), (-111, 49)],
    'North Dakota': [(-104, 49), (-97, 49), (-97, 45.9), (-104, 45.9)],
    'South Dakota': [(-104, 45.9), (-96.5, 45.9), (-96.5, 42.5), (-104, 43)],
    'Wyoming': [(-111, 45), (-104, 45), (-104, 41), (-109, 41), (-111, 44)],
    'Nebraska': [(-104, 43), (-95.5, 43), (-95.5, 40), (-102, 40), (-104, 41)],
    'Kansas': [(-102, 40), (-94.6, 40), (-94.6, 37), (-102, 37)],
    'Oklahoma': [(-103, 37), (-94.5, 37), (-94.5, 33.5), (-100, 34), (-100, 36), (-103, 36)],
    'Minnesota': [(-97, 49), (-90, 49), (-90, 43.5), (-96.5, 43.5), (-96.5, 45.9), (-97, 45.9)],
    'Iowa': [(-96.5, 43.5), (-90.2, 43.5), (-90.2, 40.4), (-95.5, 40.4), (-96.5, 42.5)],
    'Missouri': [(-95.5, 40.4), (-89, 40.4), (-89, 36), (-94.6, 36.5), (-94.6, 37), (-95.5, 40)],
    'Arkansas': [(-94.5, 36.5), (-89.5, 36.5), (-89.5, 33), (-94.5, 33)],
    'Louisiana': [(-94.5, 33), (-89, 33), (-89, 29), (-93, 29), (-94, 30)],
    'Wisconsin': [(-92.9, 47), (-87, 47), (-87, 42.5), (-90.2, 42.5), (-90.2, 43.5), (-92.9, 46)],
    'Illinois': [(-91, 42.5), (-87, 42.5), (-87, 37), (-89, 37), (-91, 40.4)],
    'Mississippi': [(-91, 35), (-88, 35), (-88, 30), (-89, 30), (-91, 31)],
    'Alabama': [(-88, 35), (-85, 35), (-85, 30.2), (-88, 30.2)],
    'Georgia': [(-85.5, 35), (-81, 35), (-81, 30.5), (-82, 30.2), (-85, 30.2)],
    'Florida': [(-87.5, 31), (-81, 31), (-80, 25), (-81, 24.5), (-82, 24.5), (-84, 29), (-87.5, 30)],
    'South Carolina': [(-83.5, 35), (-79, 35), (-79, 32.5), (-81, 32.5), (-83.5, 35)],
    'North Carolina': [(-84, 36.5), (-75.5, 36.5), (-75.5, 34), (-79, 34), (-81, 35), (-84, 35)],
    'Tennessee': [(-90, 36.5), (-82, 36.5), (-82, 35), (-89, 35), (-90, 36)],
    'Kentucky': [(-89, 37), (-82, 37), (-82, 36.5), (-89, 36.5)],
    'Virginia': [(-83, 39), (-75, 39), (-75.5, 36.5), (-84, 36.5)],
    'West Virginia': [(-82.5, 40), (-77.5, 39.5), (-79, 37.5), (-82, 37.5)],
    'Maryland': [(-79.5, 39.7), (-75, 39.7), (-75, 38), (-77, 37.5), (-79.5, 39)],
    'Delaware': [(-75.8, 39.8), (-75, 39.8), (-75, 38.5), (-75.8, 38.5)],
    'New Jersey': [(-75.5, 41.3), (-74, 41.3), (-74, 39), (-75, 39)],
    'Pennsylvania': [(-80.5, 42), (-75, 42), (-75, 39.7), (-80, 39.7)],
    'New York': [(-80, 45), (-73, 45), (-73, 40.5), (-74, 40.5), (-75, 41.3), (-75, 42), (-80, 42.5)],
    'Connecticut': [(-73.7, 42), (-71.8, 42), (-71.8, 41), (-73.7, 41)],
    'Rhode Island': [(-71.9, 42), (-71.1, 42), (-71.1, 41.3), (-71.9, 41.3)],
    'Massachusetts': [(-73.5, 42.7), (-70, 42.7), (-70, 41.5), (-71, 41.3), (-73, 42)],
    'Vermont': [(-73.4, 45), (-71.5, 45), (-71.5, 42.7), (-73, 42.7)],
    'New Hampshire': [(-72.5, 45), (-70.7, 45), (-70.7, 42.7), (-72.5, 42.7)],
    'Maine': [(-71, 47), (-67, 47), (-67, 43), (-70.7, 43), (-71, 45)],
    'Michigan': [(-90, 48), (-83, 48), (-83, 41.5), (-87, 41.5), (-87, 47)],
    'Indiana': [(-88, 41.7), (-84.8, 41.7), (-84.8, 37.8), (-88, 37.8)],
    'Ohio': [(-84.8, 42), (-80.5, 42), (-80.5, 38.5), (-84.8, 38.5)]
}

# Define high water depletion zones
high_water_zones = [
    {'name': 'California Central Valley', 'coords': [(-122.5, 35.5), (-122.5, 40), (-119, 40), (-119, 35.5)], 'level': 'Extremely High'},
    {'name': 'Southwest Desert', 'coords': [(-117, 31), (-117, 37), (-109, 37), (-109, 31)], 'level': 'Extremely High'},
    {'name': 'Texas Panhandle', 'coords': [(-103, 34), (-103, 37), (-100, 37), (-100, 34)], 'level': 'High'},
    {'name': 'Colorado River Basin', 'coords': [(-114, 35), (-114, 38), (-111, 38), (-111, 35)], 'level': 'Extremely High'},
    {'name': 'Southern California', 'coords': [(-120, 32.5), (-120, 34), (-115, 34), (-115, 32.5)], 'level': 'Extremely High'},
]

os.chdir(CODE_DIR)

# Create the enhanced map
print("\nCreating enhanced map with boundaries...")
fig, ax = plt.subplots(figsize=(20, 12))

# Draw state boundaries
for state_name, coords in state_boundaries.items():
    state_poly = MplPolygon(coords, fill=False, edgecolor='gray', linewidth=0.5, linestyle='-', alpha=0.7)
    ax.add_patch(state_poly)
    
    # Add state labels for key states
    if state_name in ['California', 'Texas', 'Arizona', 'Nevada', 'Oregon', 'Washington', 'Colorado', 'New Mexico', 'Utah', 'Florida']:
        # Calculate centroid for label placement
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        ax.text(centroid_x, centroid_y, state_name, fontsize=8, ha='center', va='center', 
                color='gray', alpha=0.7, style='italic')

# Draw US border (continental)
us_border_coords = [
    (-125, 49), (-95, 49), (-95, 49), (-67, 45), (-67, 41), (-70, 41), (-75, 35),
    (-75, 31), (-81, 25), (-84, 30), (-97, 26), (-98, 26), (-117, 32.5), (-124, 40),
    (-124, 49), (-125, 49)
]
us_border = MplPolygon(us_border_coords, fill=False, edgecolor='black', linewidth=2, linestyle='-')
ax.add_patch(us_border)

# Draw high water depletion zones
for zone in high_water_zones:
    zone_poly = MplPolygon(zone['coords'], 
                          facecolor='#8b0000' if zone['level'] == 'Extremely High' else '#ff4500',
                          alpha=0.3 if zone['level'] == 'Extremely High' else 0.25,
                          edgecolor='darkred', linewidth=1.5, linestyle='--')
    ax.add_patch(zone_poly)
    
    # Add zone label
    x_coords = [c[0] for c in zone['coords']]
    y_coords = [c[1] for c in zone['coords']]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    ax.text(centroid_x, centroid_y, zone['name'], ha='center', va='center',
           fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

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
provider_high_risk = {}

# Plot data centers
for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
    provider_high_risk[provider] = 0
    
    # Check which are in high zones
    in_high = []
    for _, dc in provider_data.iterrows():
        is_high = False
        for zone in high_water_zones:
            polygon = Polygon(zone['coords'])
            if polygon.contains(Point(dc['LOCATION_LONGITUDE'], dc['LOCATION_LATITUDE'])):
                is_high = True
                high_risk_count += 1
                provider_high_risk[provider] += 1
                break
        in_high.append(is_high)
    
    in_high = np.array(in_high)
    
    # Plot high-risk data centers with emphasis
    if in_high.any():
        high_data = provider_data[in_high]
        ax.scatter(high_data['LOCATION_LONGITUDE'], high_data['LOCATION_LATITUDE'],
                  s=180, c=provider_colors.get(provider, '#666'),
                  marker=provider_markers.get(provider, 'o'),
                  alpha=0.9, edgecolors='red', linewidth=3,
                  label=f'{provider} - High Risk ({provider_high_risk[provider]} sites)')
    
    # Plot normal data centers
    if (~in_high).any():
        normal_data = provider_data[~in_high]
        ax.scatter(normal_data['LOCATION_LONGITUDE'], normal_data['LOCATION_LATITUDE'],
                  s=100, c=provider_colors.get(provider, '#666'),
                  marker=provider_markers.get(provider, 'o'),
                  alpha=0.6, edgecolors='black', linewidth=1,
                  label=f'{provider} - Normal ({len(normal_data)} sites)')

# Set map bounds to show continental US
ax.set_xlim(-126, -66)
ax.set_ylim(24, 50)

# Labels and title
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'AI Data Centers and High Water Depletion Zones - US Map with State Boundaries\n'
           f'{high_risk_count} of {total_count} data centers ({high_risk_count/total_count*100:.1f}%) in high-risk water depletion areas',
           fontsize=16, fontweight='bold', pad=20)

# Create comprehensive legend
legend_elements = [
    mpatches.Patch(color='white', label='Water Depletion Risk Levels:'),
    mpatches.Patch(color='#8b0000', alpha=0.3, label='  Extremely High (>75% depletion)'),
    mpatches.Patch(color='#ff4500', alpha=0.25, label='  High (50-75% depletion)'),
    mpatches.Patch(color='white', label=''),  # Spacer
    mpatches.Patch(color='white', label='Geographic Boundaries:'),
    mlines.Line2D([], [], color='black', linewidth=2, label='  US Border'),
    mlines.Line2D([], [], color='gray', linewidth=0.5, label='  State Boundaries'),
    mpatches.Patch(color='white', label=''),  # Spacer
    mpatches.Patch(color='white', label='Data Centers by Provider:'),
]

# Add provider markers to legend
for provider in sorted(ai_datacenters['PROVIDER_NAME'].unique()):
    total = len(ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider])
    high = provider_high_risk[provider]
    legend_elements.append(
        plt.Line2D([0], [0], marker=provider_markers.get(provider, 'o'), color='w',
                  markerfacecolor=provider_colors.get(provider, '#666'),
                  markersize=10, label=f'  {provider}: {total} total ({high} in high-risk)')
    )

ax.legend(handles=legend_elements, loc='upper left', frameon=True,
         facecolor='white', edgecolor='black', fontsize=9)

# Add grid for reference
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

# Add coordinate reference lines for major latitudes/longitudes
for lon in [-120, -110, -100, -90, -80, -70]:
    ax.axvline(x=lon, color='lightgray', linestyle=':', alpha=0.3, linewidth=0.3)
for lat in [30, 35, 40, 45]:
    ax.axhline(y=lat, color='lightgray', linestyle=':', alpha=0.3, linewidth=0.3)

# Save high-resolution map
plt.tight_layout()
output_file = 'water_depletion_map_with_state_boundaries.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nMap saved as: {output_file}")

# Create summary statistics
print("\n" + "="*70)
print("WATER DEPLETION RISK ANALYSIS - DETAILED SUMMARY")
print("="*70)
print(f"Total AI Data Centers Analyzed: {total_count}")
print(f"Data Centers in High Water Risk Zones: {high_risk_count}")
print(f"Overall Percentage at Risk: {high_risk_count/total_count*100:.1f}%")

print("\nBreakdown by Provider:")
for provider in sorted(provider_high_risk.keys()):
    total_provider = len(ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider])
    high_risk = provider_high_risk[provider]
    pct = (high_risk/total_provider*100) if total_provider > 0 else 0
    print(f"  {provider:15} - Total: {total_provider:3d}, High-Risk: {high_risk:3d} ({pct:5.1f}%)")

print("\nHigh-Risk Water Depletion Zones:")
for zone in high_water_zones:
    print(f"  {zone['name']:25} - {zone['level']}")

# Show the map
plt.show()
print("\nMap creation complete! The visualization shows US state boundaries and high water depletion zones.")
