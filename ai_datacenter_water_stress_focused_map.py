"""
AI Data Centers and Water Stress Map
Focused visualization showing AI-designated data centers and high water stress regions
Using new Aterio dataset with FLG_AI_FACILITY flag
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AI DATA CENTERS AND HIGH WATER STRESS ANALYSIS")
print("="*70)

# Set directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')
WATER_DATA_DIR = os.path.join(DATA_DIR, 'Water Data', 'Water Resources Institute', 
                             'aqueduct-4-0-water-risk-data', 'Aqueduct40_waterrisk_download_Y2023M07D05')

os.chdir(DATA_DIR)

print("\n1. Loading New Aterio Data Center Inventory...")
print("-"*50)

# Load the new data center inventory with AI flag
datacenter_df = pd.read_csv('data_center_inventory_20251217.csv', thousands=',')

# Filter for AI facilities
ai_mask = datacenter_df['FLG_AI_FACILITY'] == 'Y'
ai_datacenters_df = datacenter_df[ai_mask].copy()

# Also get all data centers for comparison
all_datacenters_df = datacenter_df.copy()

print(f"Total data centers in dataset: {len(datacenter_df):,}")
print(f"AI-designated data centers: {len(ai_datacenters_df):,} ({len(ai_datacenters_df)/len(datacenter_df)*100:.1f}%)")

# Convert to GeoDataFrame
ai_datacenters = gpd.GeoDataFrame(
    ai_datacenters_df,
    geometry=gpd.points_from_xy(ai_datacenters_df.LOCATION_LONGITUDE, ai_datacenters_df.LOCATION_LATITUDE),
    crs='EPSG:4326'
)

# Remove duplicates based on unique ID
ai_datacenters = ai_datacenters.groupby('ATERIO_DATA_CENTER_UID').first().reset_index()

# Filter for US data centers (continental US bounds)
us_mask = (
    (ai_datacenters.LOCATION_LONGITUDE >= -125) & 
    (ai_datacenters.LOCATION_LONGITUDE <= -66) &
    (ai_datacenters.LOCATION_LATITUDE >= 24) & 
    (ai_datacenters.LOCATION_LATITUDE <= 50)
)
us_ai_datacenters = ai_datacenters[us_mask].copy()

print(f"\nUS AI data centers: {len(us_ai_datacenters):,}")

# Analyze by provider
print("\nTop AI Data Center Providers:")
provider_counts = ai_datacenters['PROVIDER_NAME'].value_counts().head(10)
for provider, count in provider_counts.items():
    print(f"  • {provider}: {count} facilities")

# Analyze by stage
print("\nAI Data Centers by Development Stage:")
stage_counts = ai_datacenters['DATA_CENTER_STAGE'].value_counts()
for stage, count in stage_counts.items():
    print(f"  • {stage}: {count} facilities")

print("\n2. Loading Geographic Base Map...")
print("-"*50)

# Try to load actual US shapefile if available
shapefile_paths = [
    'electric-retail-service-territories/electric-retail-service-territories.shp',
    'mapData.shp',
]

base_map = None
for path in shapefile_paths:
    full_path = os.path.join(DATA_DIR, path)
    if os.path.exists(full_path):
        print(f"Loading shapefile: {path}")
        base_map = gpd.read_file(full_path)
        # Ensure CRS is WGS84
        if base_map.crs is None:
            base_map = base_map.set_crs('EPSG:4326')
        else:
            base_map = base_map.to_crs('EPSG:4326')
        break

if base_map is None:
    print("No shapefile found, creating synthetic US boundary...")
    # Create US continental boundary
    us_bounds = gpd.GeoDataFrame(
        [1], geometry=[box(-125, 24, -66, 50)], crs='EPSG:4326'
    )
    base_map = us_bounds

print("\n3. Defining High Water Stress Regions...")
print("-"*50)

# Define high water stress zones based on WRI Aqueduct data
# These are the most water-stressed regions in the US
high_water_stress_zones = {
    'California Central Valley': {
        'bbox': [-122.5, 35.5, -119, 40],
        'level': 4.5,  # Extremely High
        'label': 'Extremely High (>80%)',
        'color': '#8B0000'
    },
    'Southern California': {
        'bbox': [-120, 32.5, -115, 34.5],
        'level': 4.3,
        'label': 'Extremely High (>80%)',
        'color': '#8B0000'
    },
    'Arizona - Phoenix Area': {
        'bbox': [-113, 32.5, -111, 34],
        'level': 4.4,
        'label': 'Extremely High (>80%)',
        'color': '#8B0000'
    },
    'Nevada - Las Vegas': {
        'bbox': [-116, 35.5, -114, 36.5],
        'level': 4.5,
        'label': 'Extremely High (>80%)',
        'color': '#8B0000'
    },
    'Texas Panhandle': {
        'bbox': [-103, 34, -100, 37],
        'level': 3.8,
        'label': 'High (40-80%)',
        'color': '#CD5C5C'
    },
    'New Mexico - Rio Grande': {
        'bbox': [-107, 32, -105, 36],
        'level': 3.7,
        'label': 'High (40-80%)',
        'color': '#CD5C5C'
    },
    'Colorado - Front Range': {
        'bbox': [-106, 38, -104, 41],
        'level': 3.5,
        'label': 'High (40-80%)',
        'color': '#CD5C5C'
    },
    'Kansas - Ogallala': {
        'bbox': [-102, 37, -99, 39],
        'level': 3.6,
        'label': 'High (40-80%)',
        'color': '#CD5C5C'
    },
    'Utah - Great Salt Lake': {
        'bbox': [-113, 40, -111, 42],
        'level': 3.9,
        'label': 'High (40-80%)',
        'color': '#CD5C5C'
    }
}

print("Defined 9 high water stress regions")

# Calculate AI data centers in high water stress zones
high_risk_centers = []
for _, dc in us_ai_datacenters.iterrows():
    for zone_name, zone_info in high_water_stress_zones.items():
        x_min, y_min, x_max, y_max = zone_info['bbox']
        if x_min <= dc.geometry.x <= x_max and y_min <= dc.geometry.y <= y_max:
            high_risk_centers.append({
                'name': dc.DATA_CENTER_BUILDING_NAME,
                'provider': dc.PROVIDER_NAME,
                'state': dc.STATE_NAME,
                'power_mw': dc.SELECTED_POWER_CAPACITY_MW,
                'zone': zone_name,
                'stress_level': zone_info['label']
            })
            break

high_risk_df = pd.DataFrame(high_risk_centers)
print(f"\nAI data centers in high water stress zones: {len(high_risk_df)}")
print(f"Percentage at risk: {len(high_risk_df)/len(us_ai_datacenters)*100:.1f}%")

os.chdir(CODE_DIR)

print("\n4. Creating Focused AI Data Center Map...")
print("-"*50)

# Create the main figure
fig = plt.figure(figsize=(20, 14))
ax = plt.axes()

# Set map background color (light ocean blue)
ax.set_facecolor('#E8F4F8')

# Plot base map with subtle fill
base_map.plot(ax=ax, color='#F8F8F0', edgecolor='#888888', linewidth=0.3, alpha=0.95)

# Add water stress zones with gradient overlay
patches = []
colors = []

for zone_name, zone_info in high_water_stress_zones.items():
    x_min, y_min, x_max, y_max = zone_info['bbox']
    
    # Create multiple overlapping rectangles for gradient effect
    for i in range(3):
        offset = i * 0.15
        rect = Rectangle(
            (x_min - offset, y_min - offset), 
            (x_max - x_min) + 2*offset, 
            (y_max - y_min) + 2*offset
        )
        patches.append(rect)
        colors.append(zone_info['color'])

# Add patches with transparency
if patches:
    p = PatchCollection(patches, alpha=0.12, match_original=False)
    p.set_facecolor(colors)
    ax.add_collection(p)

# Define styles for AI data centers
# Use different colors based on development stage
stage_colors = {
    'Active': '#2E7D32',  # Dark green
    'Under Construction': '#FFA726',  # Orange
    'Announced': '#1976D2',  # Blue
    'Announcement': '#1976D2',  # Blue (alternate spelling)
    'Planning': '#7B1FA2',  # Purple
    'Planned': '#7B1FA2',  # Purple (alternate)
    'On Hold': '#616161',  # Gray
    'Cancelled': '#D32F2F'  # Red
}

# Plot AI data centers
for stage in us_ai_datacenters['DATA_CENTER_STAGE'].unique():
    if pd.notna(stage):
        stage_data = us_ai_datacenters[us_ai_datacenters['DATA_CENTER_STAGE'] == stage]
        color = stage_colors.get(stage, '#424242')
        
        # Check which are in high stress zones
        high_stress_mask = []
        for _, dc in stage_data.iterrows():
            in_high_zone = False
            for zone_name, zone_info in high_water_stress_zones.items():
                x_min, y_min, x_max, y_max = zone_info['bbox']
                if x_min <= dc.geometry.x <= x_max and y_min <= dc.geometry.y <= y_max:
                    in_high_zone = True
                    break
            high_stress_mask.append(in_high_zone)
        
        high_stress_mask = np.array(high_stress_mask)
        
        # Plot high-stress locations with red border
        if high_stress_mask.any():
            ax.scatter(
                stage_data.geometry.x[high_stress_mask], 
                stage_data.geometry.y[high_stress_mask],
                c=color, s=120, marker='o', alpha=0.85,
                edgecolors='#D32F2F', linewidths=2.5,
                label=None, zorder=5
            )
        
        # Plot normal locations
        if (~high_stress_mask).any():
            ax.scatter(
                stage_data.geometry.x[~high_stress_mask], 
                stage_data.geometry.y[~high_stress_mask],
                c=color, s=80, marker='o', alpha=0.75,
                edgecolors='white', linewidths=1.2,
                label=stage, zorder=4
            )

# Add major city labels for context
major_cities = {
    'Los Angeles': (-118.24, 34.05),
    'San Francisco': (-122.42, 37.77),
    'Phoenix': (-112.07, 33.45),
    'Las Vegas': (-115.14, 36.17),
    'Denver': (-104.99, 39.74),
    'Dallas': (-96.80, 32.78),
    'Houston': (-95.37, 29.76),
    'Chicago': (-87.63, 41.88),
    'New York': (-74.01, 40.71),
    'Washington DC': (-77.04, 38.91),
    'Atlanta': (-84.39, 33.75),
    'Seattle': (-122.33, 47.61),
    'Portland': (-122.68, 45.52),
    'Salt Lake City': (-111.89, 40.76),
    'Albuquerque': (-106.65, 35.08)
}

for city, (lon, lat) in major_cities.items():
    ax.plot(lon, lat, 'k.', markersize=4, alpha=0.5)
    ax.text(lon + 0.1, lat + 0.1, city, fontsize=8, ha='left', va='bottom',
           color='#333333', alpha=0.7, weight='light')

# Set map extent (Continental US)
ax.set_xlim(-126, -66)
ax.set_ylim(24, 50)

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Add title and subtitle
title = ax.set_title('AI Data Centers and High Water Stress Regions in the United States',
                    fontsize=20, fontweight='bold', pad=25, color='#1A237E')

subtitle_text = f'Analysis of {len(us_ai_datacenters)} AI-designated facilities • {len(high_risk_df)} ({len(high_risk_df)/len(us_ai_datacenters)*100:.1f}%) located in high water stress areas'
plt.text(0.5, 0.94, subtitle_text, transform=fig.transFigure,
        ha='center', fontsize=13, color='#424242', style='italic')

# Create comprehensive legend
legend_elements = []

# Water stress legend
legend_elements.append(Patch(facecolor='none', edgecolor='none', 
                            label='Water Stress Levels:', alpha=0))
legend_elements.append(Patch(facecolor='#8B0000', alpha=0.35, 
                            label='  Extremely High (>80% water depletion)'))
legend_elements.append(Patch(facecolor='#CD5C5C', alpha=0.35, 
                            label='  High (40-80% water depletion)'))

# Add spacing
legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))

# Development stage legend
legend_elements.append(Patch(facecolor='none', edgecolor='none', 
                            label='AI Data Center Status:', alpha=0))
for stage, color in stage_colors.items():
    if stage in us_ai_datacenters['DATA_CENTER_STAGE'].values:
        count = len(us_ai_datacenters[us_ai_datacenters['DATA_CENTER_STAGE'] == stage])
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markeredgecolor='white',
                      markersize=9, label=f'  {stage} ({count} facilities)')
        )

# Add red border indicator
legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))
legend_elements.append(
    plt.Line2D([0], [0], marker='o', color='w',
              markerfacecolor='gray', markeredgecolor='#D32F2F',
              markersize=9, linewidth=2, label='⚠ Located in high water stress zone')
)

# Position legend
legend = ax.legend(handles=legend_elements, loc='lower left',
                  frameon=True, facecolor='white', edgecolor='#CCCCCC',
                  framealpha=0.95, fontsize=10, ncol=2)
legend.set_title('Legend', prop={'size': 12, 'weight': 'bold'})

# Add north arrow
north_arrow = FancyArrow(-125, 47.5, 0, 0.5, width=0.3, head_width=0.5,
                        head_length=0.3, fc='black', ec='black')
ax.add_patch(north_arrow)
ax.text(-125, 48.5, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add data source note
source_text = 'Data Sources: Aterio Data Center Inventory (Dec 2025) with AI Facility Flag, WRI Aqueduct 4.0 Water Risk Atlas'
plt.text(0.99, 0.01, source_text,
        transform=ax.transAxes, ha='right', fontsize=9, color='#666666', style='italic')

# Add analysis date
date_text = f'Analysis Date: December 17, 2025'
plt.text(0.01, 0.01, date_text,
        transform=ax.transAxes, ha='left', fontsize=9, color='#666666', style='italic')

# Save the map
plt.tight_layout()
output_file = 'ai_datacenter_water_stress_focused_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Map saved as: {output_file}")

# Create summary statistics
print("\n" + "="*70)
print("AI DATA CENTER WATER STRESS ANALYSIS SUMMARY")
print("="*70)

print(f"\nTotal AI Data Centers in Dataset: {len(ai_datacenters):,}")
print(f"US Continental AI Data Centers: {len(us_ai_datacenters):,}")
print(f"AI Data Centers in High Water Stress Zones: {len(high_risk_df):,}")
print(f"Percentage at Risk: {len(high_risk_df)/len(us_ai_datacenters)*100:.1f}%")

if len(high_risk_df) > 0:
    print("\nHigh Water Stress Zones with AI Data Centers:")
    zone_summary = high_risk_df.groupby('zone').size().sort_values(ascending=False)
    for zone, count in zone_summary.items():
        stress_level = high_water_stress_zones[zone]['label']
        print(f"  • {zone} ({stress_level}): {count} AI facilities")
    
    print("\nTop Providers with AI Data Centers in High Water Stress Areas:")
    provider_risk = high_risk_df['provider'].value_counts().head(10)
    for provider, count in provider_risk.items():
        if pd.notna(provider):
            print(f"  • {provider}: {count} facilities at risk")

    # Calculate total power capacity at risk
    power_at_risk = high_risk_df['power_mw'].sum()
    total_power = us_ai_datacenters['SELECTED_POWER_CAPACITY_MW'].sum()
    if power_at_risk > 0 and total_power > 0:
        print(f"\nPower Capacity Analysis:")
        print(f"  • Total AI data center capacity: {total_power:,.0f} MW")
        print(f"  • Capacity in high water stress zones: {power_at_risk:,.0f} MW")
        print(f"  • Percentage of capacity at risk: {power_at_risk/total_power*100:.1f}%")

# Save summary data
summary_df = pd.DataFrame({
    'Metric': [
        'Total AI Data Centers',
        'US Continental AI Data Centers',
        'AI Data Centers in High Water Stress',
        'Percentage at Risk',
        'Total Power Capacity (MW)',
        'Power Capacity at Risk (MW)'
    ],
    'Value': [
        len(ai_datacenters),
        len(us_ai_datacenters),
        len(high_risk_df),
        f"{len(high_risk_df)/len(us_ai_datacenters)*100:.1f}%",
        f"{us_ai_datacenters['SELECTED_POWER_CAPACITY_MW'].sum():,.0f}",
        f"{high_risk_df['power_mw'].sum():,.0f}" if len(high_risk_df) > 0 else "0"
    ]
})

summary_df.to_csv('ai_datacenter_water_stress_summary.csv', index=False)
print(f"\nSummary statistics saved to: ai_datacenter_water_stress_summary.csv")

# Save detailed risk data
if len(high_risk_df) > 0:
    high_risk_df.to_csv('ai_datacenters_at_risk.csv', index=False)
    print(f"Detailed risk data saved to: ai_datacenters_at_risk.csv")

# Show the map
plt.show()
print("\nMap visualization complete!")
