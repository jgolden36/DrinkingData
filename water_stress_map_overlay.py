"""
Water Stress Map Overlay - AI Data Centers and High Water Depletion Regions
Creates a detailed map showing AI data centers overlaid on water depletion zones
with emphasis on high-stress areas.
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, Rectangle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("WATER DEPLETION MAP WITH AI DATA CENTER OVERLAY")
print("="*70)

# Set directories
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

def load_data():
    """Load AI data centers and water stress data."""
    print("\n1. Loading Data...")
    print("-"*50)
    
    os.chdir(DATA_DIR)
    
    # Load AI models
    AI_df = pd.read_csv('Epoch Database - Notable Models.csv', thousands=',')
    AI_df["Publication date"] = pd.to_datetime(AI_df["Publication date"])
    AI_df[['primaryDataCenterProvider', 'secondaryDataCenterProvider']] = AI_df['Organization'].apply(
        lambda x: pd.Series(map_organization_to_provider(x))
    )
    AI_df = AI_df[(AI_df['Frontier model'] == 'checked') & (AI_df['primaryDataCenterProvider'].notna())]
    
    # Load data centers
    dataCenterdf = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')
    dataCenterdf['geometry'] = dataCenterdf.apply(
        lambda row: Point(row['LOCATION_LONGITUDE'], row['LOCATION_LATITUDE']), axis=1
    )
    dataCenters_gdf = gpd.GeoDataFrame(dataCenterdf, geometry='geometry', crs='EPSG:4326')
    
    # Get AI-related data centers
    ai_datacenters = dataCenters_gdf.merge(
        AI_df[['primaryDataCenterProvider', 'Model', 'Organization']].drop_duplicates(),
        left_on='PROVIDER_NAME',
        right_on='primaryDataCenterProvider',
        how='inner'
    )
    
    # Remove duplicates
    ai_datacenters_unique = ai_datacenters.groupby('ATERIO_DATA_CENTER_UID').first().reset_index()
    
    print(f"Loaded {len(ai_datacenters_unique)} unique AI data centers")
    
    # Load water stress data
    csv_path = os.path.join(WATER_DATA_DIR, 'CVS', 'Aqueduct40_baseline_annual_y2023m07d05.csv')
    
    print("Loading water stress regions...")
    
    # Read and process water data by US states
    water_data = []
    chunk_size = 100000
    
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        # Focus on US data
        us_data = chunk[chunk['name_0'] == 'United States of America']
        if len(us_data) > 0:
            state_summary = us_data.groupby('name_1').agg({
                'bwd_score': 'mean',
                'bwd_label': lambda x: x.mode()[0] if len(x) > 0 else 'No Data',
                'bws_score': 'mean',
                'bws_label': lambda x: x.mode()[0] if len(x) > 0 else 'No Data'
            })
            water_data.append(state_summary)
        
        if i >= 10:  # Process first 1M rows
            break
    
    if water_data:
        water_stress_by_state = pd.concat(water_data).groupby(level=0).mean()
        water_stress_by_state['bwd_label'] = water_stress_by_state.index.map(
            lambda x: us_data[us_data['name_1'] == x]['bwd_label'].mode()[0] 
            if len(us_data[us_data['name_1'] == x]) > 0 else 'No Data'
        )
    else:
        water_stress_by_state = pd.DataFrame()
    
    print(f"Loaded water stress data for {len(water_stress_by_state)} US states")
    
    return ai_datacenters_unique, water_stress_by_state

def create_water_depletion_zones():
    """Create synthetic water depletion zones for visualization."""
    
    # Define high water depletion zones in the US (based on known water-stressed regions)
    high_depletion_zones = [
        # California Central Valley
        {'name': 'California Central Valley', 
         'coords': [(-122.5, 35.5), (-122.5, 40), (-119, 40), (-119, 35.5)],
         'level': 'Extremely High'},
        
        # Southwest (Arizona, Nevada, Southern California)
        {'name': 'Southwest Desert', 
         'coords': [(-117, 31), (-117, 37), (-109, 37), (-109, 31)],
         'level': 'Extremely High'},
        
        # Texas Panhandle
        {'name': 'Texas Panhandle', 
         'coords': [(-103, 34), (-103, 37), (-100, 37), (-100, 34)],
         'level': 'High'},
        
        # Colorado River Basin
        {'name': 'Colorado River Basin', 
         'coords': [(-114, 35), (-114, 38), (-111, 38), (-111, 35)],
         'level': 'Extremely High'},
        
        # Southern Great Plains
        {'name': 'Southern Great Plains', 
         'coords': [(-102, 32), (-102, 35), (-97, 35), (-97, 32)],
         'level': 'High'},
        
        # Central California
        {'name': 'Central California', 
         'coords': [(-124, 36), (-124, 38.5), (-120, 38.5), (-120, 36)],
         'level': 'High'},
    ]
    
    # Medium depletion zones
    medium_depletion_zones = [
        # Southeast
        {'name': 'Southeast', 
         'coords': [(-88, 30), (-88, 35), (-75, 35), (-75, 30)],
         'level': 'Medium'},
        
        # Midwest
        {'name': 'Midwest', 
         'coords': [(-96, 40), (-96, 44), (-88, 44), (-88, 40)],
         'level': 'Medium'},
        
        # Mid-Atlantic
        {'name': 'Mid-Atlantic', 
         'coords': [(-82, 36), (-82, 41), (-74, 41), (-74, 36)],
         'level': 'Medium'},
    ]
    
    return high_depletion_zones + medium_depletion_zones

def create_detailed_map(ai_datacenters, water_zones):
    """Create a detailed map with water depletion zones and data centers."""
    print("\n2. Creating Detailed Water Depletion Map...")
    print("-"*50)
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Color scheme for water depletion levels
    depletion_colors = {
        'Extremely High': '#8b0000',  # Dark red
        'High': '#ff4500',            # Orange red
        'Medium': '#ffa500',          # Orange
        'Low': '#90ee90',             # Light green
        'No Data': '#f0f0f0'          # Light gray
    }
    
    # Draw water depletion zones
    for zone in water_zones:
        polygon = Polygon(zone['coords'])
        color = depletion_colors.get(zone['level'], '#f0f0f0')
        alpha = 0.4 if zone['level'] == 'Extremely High' else 0.3
        
        # Create patch
        x, y = polygon.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha, edgecolor='black', linewidth=1, linestyle='--')
        
        # Add zone label
        centroid_x = sum([c[0] for c in zone['coords']]) / len(zone['coords'])
        centroid_y = sum([c[1] for c in zone['coords']]) / len(zone['coords'])
        ax.text(centroid_x, centroid_y, zone['name'], 
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Provider colors and markers
    provider_styles = {
        'Google': {'color': '#4285F4', 'marker': 'o', 'size': 100},
        'Microsoft': {'color': '#00A4EF', 'marker': 's', 'size': 100},
        'Facebook': {'color': '#1877F2', 'marker': '^', 'size': 100},
        'Apple Inc.': {'color': '#555555', 'marker': 'D', 'size': 100},
        'Amazon AWS': {'color': '#FF9900', 'marker': 'v', 'size': 100}
    }
    
    # Plot data centers
    for provider in ai_datacenters['PROVIDER_NAME'].unique():
        provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
        style = provider_styles.get(provider, {'color': '#666666', 'marker': 'o', 'size': 80})
        
        # Check which data centers are in high depletion zones
        in_high_zone = []
        for _, dc in provider_data.iterrows():
            is_high = False
            for zone in water_zones:
                if zone['level'] in ['High', 'Extremely High']:
                    polygon = Polygon(zone['coords'])
                    if polygon.contains(Point(dc.geometry.x, dc.geometry.y)):
                        is_high = True
                        break
            in_high_zone.append(is_high)
        
        # Plot data centers in high zones with different edge
        high_zone_mask = np.array(in_high_zone)
        
        if high_zone_mask.any():
            ax.scatter(provider_data.geometry.x[high_zone_mask], 
                      provider_data.geometry.y[high_zone_mask],
                      s=style['size']*1.5, 
                      c=style['color'],
                      marker=style['marker'],
                      alpha=0.9,
                      edgecolors='red',
                      linewidth=2,
                      label=f'{provider} (High Risk)')
        
        if (~high_zone_mask).any():
            ax.scatter(provider_data.geometry.x[~high_zone_mask], 
                      provider_data.geometry.y[~high_zone_mask],
                      s=style['size'], 
                      c=style['color'],
                      marker=style['marker'],
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=1,
                      label=f'{provider}')
    
    # Set map bounds (Continental US)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 49)
    
    # Add title and labels
    ax.set_title('AI Data Centers and Water Depletion Zones\nHigh-Risk Areas Highlighted', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Create custom legend
    legend_elements = []
    
    # Water depletion levels
    legend_elements.append(mpatches.Patch(color='white', label='Water Depletion Levels:'))
    legend_elements.append(mpatches.Patch(color='#8b0000', alpha=0.4, label='Extremely High (>75%)'))
    legend_elements.append(mpatches.Patch(color='#ff4500', alpha=0.3, label='High (50-75%)'))
    legend_elements.append(mpatches.Patch(color='#ffa500', alpha=0.3, label='Medium (25-50%)'))
    
    # Add provider legend
    legend_elements.append(mpatches.Patch(color='white', label='\nData Center Providers:'))
    for provider in ai_datacenters['PROVIDER_NAME'].unique():
        style = provider_styles.get(provider, {'color': '#666666', 'marker': 'o'})
        n_centers = len(ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider])
        legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], color='w', 
                                         markerfacecolor=style['color'], markersize=10,
                                         label=f'{provider} ({n_centers} sites)'))
    
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
             facecolor='white', edgecolor='black', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add state boundaries (simplified)
    ax.axvline(x=-120, color='gray', linestyle=':', alpha=0.5)  # California border
    ax.axvline(x=-114, color='gray', linestyle=':', alpha=0.5)  # Arizona border
    ax.axvline(x=-109, color='gray', linestyle=':', alpha=0.5)  # New Mexico border
    ax.axvline(x=-104, color='gray', linestyle=':', alpha=0.5)  # Colorado border
    ax.axhline(y=42, color='gray', linestyle=':', alpha=0.5)    # Northern border
    ax.axhline(y=37, color='gray', linestyle=':', alpha=0.5)    # Southern tier
    
    plt.tight_layout()
    
    # Save map
    output_file = 'water_depletion_datacenter_overlay_map.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig

def create_risk_analysis(ai_datacenters, water_zones):
    """Analyze data centers in high-risk water zones."""
    print("\n3. Analyzing Data Centers in High-Risk Zones...")
    print("-"*50)
    
    # Provider colors and markers (needed for visualization)
    provider_styles = {
        'Google': {'color': '#4285F4', 'marker': 'o', 'size': 100},
        'Microsoft': {'color': '#00A4EF', 'marker': 's', 'size': 100},
        'Facebook': {'color': '#1877F2', 'marker': '^', 'size': 100},
        'Apple Inc.': {'color': '#555555', 'marker': 'D', 'size': 100},
        'Amazon AWS': {'color': '#FF9900', 'marker': 'v', 'size': 100}
    }
    
    # Count data centers in each risk level
    risk_counts = {'Extremely High': {}, 'High': {}, 'Medium': {}, 'Low': {}}
    
    for provider in ai_datacenters['PROVIDER_NAME'].unique():
        provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
        
        for level in risk_counts:
            count = 0
            for _, dc in provider_data.iterrows():
                for zone in water_zones:
                    if zone['level'] == level:
                        polygon = Polygon(zone['coords'])
                        if polygon.contains(Point(dc.geometry.x, dc.geometry.y)):
                            count += 1
                            break
            risk_counts[level][provider] = count
    
    # Create summary DataFrame
    risk_df = pd.DataFrame(risk_counts).T
    risk_df = risk_df.fillna(0).astype(int)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stacked bar chart by provider
    ax1 = axes[0]
    risk_df.T.plot(kind='bar', stacked=True, ax=ax1,
                   color=['#8b0000', '#ff4500', '#ffa500', '#90ee90'])
    ax1.set_title('Data Centers by Water Depletion Risk Level', fontweight='bold')
    ax1.set_xlabel('Provider')
    ax1.set_ylabel('Number of Data Centers')
    ax1.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart of high-risk data centers
    ax2 = axes[1]
    high_risk_total = risk_df.loc['Extremely High'] + risk_df.loc['High']
    high_risk_total = high_risk_total[high_risk_total > 0]
    
    if len(high_risk_total) > 0:
        colors = [provider_styles.get(p, {'color': '#666666'})['color'] for p in high_risk_total.index]
        wedges, texts, autotexts = ax2.pie(high_risk_total, labels=high_risk_total.index, 
                                           colors=colors, autopct='%1.1f%%',
                                           startangle=90)
        ax2.set_title(f'Distribution of {high_risk_total.sum()} Data Centers\nin High/Extreme Water Risk Areas', 
                     fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No data centers in high-risk areas', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('High-Risk Data Centers', fontweight='bold')
    
    plt.suptitle('Water Depletion Risk Analysis for AI Data Centers', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = 'water_depletion_risk_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Print summary
    print("\nRisk Analysis Summary:")
    print("="*40)
    print(risk_df)
    print("\nTotal data centers by risk level:")
    print(risk_df.sum(axis=1))
    
    # Save to CSV
    risk_df.to_csv('datacenter_water_risk_summary.csv')
    print("\nSaved: datacenter_water_risk_summary.csv")
    
    return fig, risk_df

def main():
    """Main execution."""
    try:
        os.chdir(CODE_DIR)
        
        # Load data
        ai_datacenters, water_stress_by_state = load_data()
        
        # Create water depletion zones
        water_zones = create_water_depletion_zones()
        
        # Create main overlay map
        map_fig = create_detailed_map(ai_datacenters, water_zones)
        
        # Create risk analysis
        risk_fig, risk_summary = create_risk_analysis(ai_datacenters, water_zones)
        
        print("\n" + "="*70)
        print("MAP CREATION COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("1. water_depletion_datacenter_overlay_map.png - Main overlay map")
        print("2. water_depletion_risk_analysis.png - Risk analysis charts")
        print("3. datacenter_water_risk_summary.csv - Risk summary data")
        
        plt.show()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
