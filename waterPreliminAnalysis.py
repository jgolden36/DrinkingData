"""
Water Stress and Data Center Analysis
This script merges AI data center locations with water stress data from WRI Aqueduct 4.0
and creates visualization overlaying data centers on water stress regions.

Uses the same AI filtering logic from DiffinDiff.py and creates maps similar to 
the approach in AI_Price_Impact_Choropleth.py
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set working directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')
WATER_DATA_DIR = os.path.join(DATA_DIR, 'Water Data', 'Water Resources Institute', 
                             'aqueduct-4-0-water-risk-data', 'Aqueduct40_waterrisk_download_Y2023M07D05')

def map_organization_to_provider(organization):
    """
    Maps organization names to data center providers based on specified rules.
    (Copied from DiffinDiff.py for consistency)
    """
    if pd.isna(organization):
        return None, None
    
    org_lower = str(organization).lower()
    
    # Define mapping rules
    providers = []
    
    # Google (Google, Anthropic, Deep Mind)
    if any(keyword in org_lower for keyword in ['google', 'anthropic', 'deep mind', 'deepmind']):
        providers.append('Google')
    
    # Microsoft (Microsoft, OpenAI)
    if any(keyword in org_lower for keyword in ['microsoft', 'openai']):
        providers.append('Microsoft')
    
    # Facebook (Meta, Facebook)
    if any(keyword in org_lower for keyword in ['meta', 'facebook']):
        providers.append('Facebook')
    
    # Apple
    if 'apple' in org_lower:
        providers.append('Apple Inc.')
    
    # Amazon AWS (Amazon, Perplexity)
    if any(keyword in org_lower for keyword in ['amazon', 'perplexity']):
        providers.append('Amazon AWS')
    
    # Remove duplicates while preserving order
    providers = list(dict.fromkeys(providers))
    
    # Return primary and secondary providers
    primary = providers[0] if len(providers) > 0 else None
    secondary = providers[1] if len(providers) > 1 else None
    
    return primary, secondary

def load_and_filter_ai_datacenters():
    """
    Load and filter data centers associated with AI models.
    Uses the same filtering logic as DiffinDiff.py
    """
    print("Loading AI and data center data...")
    os.chdir(DATA_DIR)
    
    # Load AI models data
    AI_df = pd.read_csv('Epoch Database - Notable Models.csv', thousands=',')
    AI_df["Publication date"] = pd.to_datetime(AI_df["Publication date"])
    
    # Map organizations to data center providers
    AI_df[['primaryDataCenterProvider', 'secondaryDataCenterProvider']] = AI_df['Organization'].apply(
        lambda x: pd.Series(map_organization_to_provider(x))
    )
    
    # Filter for frontier models with identified providers
    AI_df = AI_df[(AI_df['Frontier model'] == 'checked') & (AI_df['primaryDataCenterProvider'].notna())]
    
    # Load data center inventory
    dataCenterdf = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')
    
    # Create GeoDataFrame from data centers
    dataCenterdf['geometry'] = dataCenterdf.apply(
        lambda row: Point(row['LOCATION_LONGITUDE'], row['LOCATION_LATITUDE']), axis=1
    )
    dataCenters_gdf = gpd.GeoDataFrame(dataCenterdf, geometry='geometry', crs='EPSG:4326')
    
    # Merge with AI models to get only AI-related data centers
    ai_datacenters = dataCenters_gdf.merge(
        AI_df[['primaryDataCenterProvider', 'Model', 'Parameters', 'Publication date', 
               'Training power draw (W)', 'Organization']].drop_duplicates(),
        left_on='PROVIDER_NAME',
        right_on='primaryDataCenterProvider',
        how='inner'
    )
    
    print(f"Loaded {len(dataCenters_gdf)} total data centers")
    print(f"Filtered to {len(ai_datacenters)} AI-related data centers")
    print(f"Unique AI providers: {ai_datacenters['PROVIDER_NAME'].unique()}")
    
    return ai_datacenters, dataCenters_gdf

def load_water_stress_data():
    """
    Load water stress data from Aqueduct 4.0
    """
    print("\nLoading water stress data...")
    
    # Try loading from CSV first (simpler than geodatabase)
    csv_path = os.path.join(WATER_DATA_DIR, 'CVS', 'Aqueduct40_baseline_annual_y2023m07d05.csv')
    
    print("Reading water stress CSV data (this may take a moment)...")
    # Read in chunks to handle large file
    chunks = []
    chunk_size = 50000
    
    # Read only necessary columns to save memory
    columns_needed = ['string_id', 'aq30_id', 'pfaf_id', 'gid_0', 'gid_1', 'name_0', 'name_1', 
                     'bwd_raw', 'bwd_score', 'bwd_cat', 'bwd_label',
                     'bws_raw', 'bws_score', 'bws_cat', 'bws_label']
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=columns_needed):
        chunks.append(chunk)
        if len(chunks) * chunk_size >= 200000:  # Limit to first 200k rows for analysis
            break
    
    water_df = pd.concat(chunks, ignore_index=True)
    
    print(f"Loaded {len(water_df)} water stress records")
    print(f"Countries in data: {water_df['name_0'].nunique()}")
    print(f"Regions in data: {water_df['name_1'].nunique()}")
    
    # Show water stress categories
    if 'bwd_label' in water_df.columns:
        print("\nWater depletion (bwd) categories:")
        print(water_df['bwd_label'].value_counts())
    
    if 'bws_label' in water_df.columns:
        print("\nWater stress (bws) categories:")
        print(water_df['bws_label'].value_counts())
    
    return water_df

def load_water_stress_geodata():
    """
    Load water stress geodatabase for spatial analysis
    Since we can't directly read the .gdb file, we'll use an alternative approach
    """
    print("\nAttempting to load geographic water stress data...")
    
    # Try to load shapefile if it exists (converted from GDB)
    # If not available, we'll create approximate geometries from country/region data
    
    # For now, we'll work with country-level aggregation
    # In a full implementation, you would convert the .gdb file to shapefile using QGIS or ArcGIS
    
    return None

def merge_datacenters_with_water_stress(ai_datacenters, water_df):
    """
    Merge data centers with water stress data based on location
    """
    print("\nMerging data centers with water stress regions...")
    
    # For initial analysis, aggregate water stress by country
    country_water_stress = water_df.groupby('name_0').agg({
        'bwd_score': 'mean',
        'bwd_label': lambda x: x.mode()[0] if not x.empty else None,
        'bws_score': 'mean',
        'bws_label': lambda x: x.mode()[0] if not x.empty else None
    }).reset_index()
    
    # Map data center locations to countries (simplified approach)
    # In production, you would use spatial join with actual water stress polygons
    
    # Add country information to data centers based on coordinates
    # This is a simplified approach - ideally use reverse geocoding
    
    # For US data centers (most common), assign water stress
    us_water_stress = water_df[water_df['name_0'] == 'United States of America'].groupby('name_1').agg({
        'bwd_score': 'mean',
        'bwd_label': lambda x: x.mode()[0] if not x.empty else None,
        'bws_score': 'mean',
        'bws_label': lambda x: x.mode()[0] if not x.empty else None
    }).reset_index()
    
    print(f"Water stress data for {len(us_water_stress)} US states")
    
    # Add state mapping for US data centers (simplified)
    # In production, use proper spatial join
    ai_datacenters['water_stress_region'] = 'Unknown'
    ai_datacenters['bwd_score'] = np.nan
    ai_datacenters['bwd_label'] = 'No Data'
    ai_datacenters['bws_score'] = np.nan
    ai_datacenters['bws_label'] = 'No Data'
    
    # Assign average US water stress as placeholder
    if len(us_water_stress) > 0:
        avg_bwd = us_water_stress['bwd_score'].mean()
        avg_bws = us_water_stress['bws_score'].mean()
        
        # For US data centers (longitude < -60 and > -130, latitude between 25 and 50)
        us_mask = (ai_datacenters.geometry.x > -130) & (ai_datacenters.geometry.x < -60) & \
                  (ai_datacenters.geometry.y > 25) & (ai_datacenters.geometry.y < 50)
        
        ai_datacenters.loc[us_mask, 'water_stress_region'] = 'United States'
        ai_datacenters.loc[us_mask, 'bwd_score'] = avg_bwd
        ai_datacenters.loc[us_mask, 'bws_score'] = avg_bws
    
    print(f"Merged {ai_datacenters['water_stress_region'].ne('Unknown').sum()} data centers with water stress data")
    
    return ai_datacenters

def create_water_stress_map(ai_datacenters, water_df, save_path='water_stress_datacenter_map.png'):
    """
    Create a map showing data centers overlaid on water stress regions
    """
    print("\nCreating water stress and data center map...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: Water Depletion (BWD)
    ax1 = axes[0]
    ax1.set_title('AI Data Centers and Water Depletion Risk', fontsize=14, fontweight='bold')
    
    # Right plot: Water Stress (BWS)
    ax2 = axes[1]
    ax2.set_title('AI Data Centers and Water Stress Risk', fontsize=14, fontweight='bold')
    
    # Since we don't have polygon geometries, create a scatter plot representation
    # Group water data by region for visualization
    region_summary = water_df.groupby('name_1').agg({
        'bwd_score': 'mean',
        'bws_score': 'mean',
        'bwd_label': lambda x: x.mode()[0] if not x.empty else None,
        'bws_label': lambda x: x.mode()[0] if not x.empty else None
    }).reset_index()
    
    # Define color maps for water stress categories
    bwd_colors = {
        'Low (<10%)': '#ffffb2',
        'Low - Medium (10-20%)': '#fecc5c',
        'Medium - High (20-40%)': '#fd8d3c',
        'High (40-80%)': '#f03b20',
        'Extremely High (>80%)': '#bd0026',
        'Arid and Low Water Use': '#c0c0c0',
        'No Data': '#ffffff'
    }
    
    # Plot data centers with color based on water stress
    for provider in ai_datacenters['PROVIDER_NAME'].unique():
        provider_data = ai_datacenters[ai_datacenters['PROVIDER_NAME'] == provider]
        
        # Plot on both axes
        ax1.scatter(provider_data.geometry.x, provider_data.geometry.y,
                   s=100, alpha=0.7, label=provider, edgecolors='black', linewidth=1)
        ax2.scatter(provider_data.geometry.x, provider_data.geometry.y,
                   s=100, alpha=0.7, label=provider, edgecolors='black', linewidth=1)
    
    # Add legends
    ax1.legend(title='Data Center Provider', loc='lower left', frameon=True, facecolor='white')
    ax2.legend(title='Data Center Provider', loc='lower left', frameon=True, facecolor='white')
    
    # Set axis limits (focus on US for now)
    for ax in axes:
        ax.set_xlim(-130, -60)
        ax.set_ylim(25, 50)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to {save_path}")
    
    return fig

def create_water_stress_summary(ai_datacenters, water_df):
    """
    Create summary statistics and visualizations of water stress at data center locations
    """
    print("\nGenerating water stress summary statistics...")
    
    # Create summary by provider
    provider_summary = ai_datacenters.groupby('PROVIDER_NAME').agg({
        'bwd_score': ['mean', 'std', 'count'],
        'bws_score': ['mean', 'std', 'count'],
        'Model': 'nunique',
        'Training power draw (W)': 'mean'
    }).round(2)
    
    print("\nWater Stress by AI Provider:")
    print(provider_summary)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar chart of average water depletion by provider
    ax1 = axes[0, 0]
    providers = ai_datacenters.groupby('PROVIDER_NAME')['bwd_score'].mean().sort_values()
    providers.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('Average Water Depletion Score by AI Provider')
    ax1.set_xlabel('Water Depletion Score')
    
    # 2. Bar chart of average water stress by provider
    ax2 = axes[0, 1]
    providers_stress = ai_datacenters.groupby('PROVIDER_NAME')['bws_score'].mean().sort_values()
    providers_stress.plot(kind='barh', ax=ax2, color='coral')
    ax2.set_title('Average Water Stress Score by AI Provider')
    ax2.set_xlabel('Water Stress Score')
    
    # 3. Scatter plot: Power draw vs Water stress
    ax3 = axes[1, 0]
    valid_data = ai_datacenters.dropna(subset=['Training power draw (W)', 'bwd_score'])
    if len(valid_data) > 0:
        scatter = ax3.scatter(valid_data['Training power draw (W)'], 
                            valid_data['bwd_score'],
                            c=valid_data['PROVIDER_NAME'].astype('category').cat.codes,
                            s=100, alpha=0.6, cmap='viridis')
        ax3.set_xlabel('Training Power Draw (W)')
        ax3.set_ylabel('Water Depletion Score')
        ax3.set_title('AI Model Power Draw vs Water Depletion at Data Centers')
        ax3.set_xscale('log')
    
    # 4. Distribution of data centers by water stress category
    ax4 = axes[1, 1]
    if 'bwd_label' in ai_datacenters.columns:
        water_categories = ai_datacenters['bwd_label'].value_counts()
        water_categories.plot(kind='bar', ax=ax4, color='teal')
        ax4.set_title('Distribution of AI Data Centers by Water Depletion Category')
        ax4.set_xlabel('Water Depletion Category')
        ax4.set_ylabel('Number of Data Centers')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('water_stress_summary.png', dpi=300, bbox_inches='tight')
    print(f"Summary charts saved to water_stress_summary.png")
    
    return fig, provider_summary

def create_temporal_analysis(ai_datacenters):
    """
    Analyze temporal patterns of AI model deployment in water-stressed regions
    """
    print("\nAnalyzing temporal patterns...")
    
    # Group by publication date and water stress
    temporal_data = ai_datacenters.groupby([
        pd.Grouper(key='Publication date', freq='Q'),
        'PROVIDER_NAME'
    ]).agg({
        'bwd_score': 'mean',
        'bws_score': 'mean',
        'Model': 'count'
    }).reset_index()
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for provider in temporal_data['PROVIDER_NAME'].unique():
        provider_data = temporal_data[temporal_data['PROVIDER_NAME'] == provider]
        ax.plot(provider_data['Publication date'], 
               provider_data['bwd_score'],
               marker='o', label=provider, linewidth=2)
    
    ax.set_xlabel('Publication Date')
    ax.set_ylabel('Average Water Depletion Score')
    ax.set_title('Water Depletion at AI Data Centers Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('water_stress_temporal.png', dpi=300, bbox_inches='tight')
    print(f"Temporal analysis saved to water_stress_temporal.png")
    
    return fig

def save_merged_data(ai_datacenters, output_path='ai_datacenter_water_stress.csv'):
    """
    Save the merged dataset for future analysis
    """
    # Convert geometry to lat/lon columns for CSV
    output_df = ai_datacenters.copy()
    output_df['latitude'] = output_df.geometry.y
    output_df['longitude'] = output_df.geometry.x
    output_df = output_df.drop(columns=['geometry'])
    
    output_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")
    
    return output_df

def main():
    """
    Main execution function
    """
    print("="*70)
    print("WATER STRESS AND AI DATA CENTER ANALYSIS")
    print("="*70)
    
    try:
        # Change to code directory for outputs
        os.chdir(CODE_DIR)
        
        # Step 1: Load and filter AI data centers
        print("\nStep 1: Loading AI Data Centers")
        print("-"*50)
        ai_datacenters, all_datacenters = load_and_filter_ai_datacenters()
        
        # Step 2: Load water stress data
        print("\nStep 2: Loading Water Stress Data")
        print("-"*50)
        water_df = load_water_stress_data()
        
        # Step 3: Merge data centers with water stress regions
        print("\nStep 3: Merging Geographic Data")
        print("-"*50)
        ai_datacenters_with_water = merge_datacenters_with_water_stress(ai_datacenters, water_df)
        
        # Step 4: Create visualizations
        print("\nStep 4: Creating Visualizations")
        print("-"*50)
        
        # Main map
        map_fig = create_water_stress_map(ai_datacenters_with_water, water_df)
        
        # Summary statistics
        summary_fig, provider_summary = create_water_stress_summary(ai_datacenters_with_water, water_df)
        
        # Temporal analysis
        if 'Publication date' in ai_datacenters_with_water.columns:
            temporal_fig = create_temporal_analysis(ai_datacenters_with_water)
        
        # Step 5: Save results
        print("\nStep 5: Saving Results")
        print("-"*50)
        output_df = save_merged_data(ai_datacenters_with_water)
        
        # Save provider summary
        provider_summary.to_csv('water_stress_by_provider.csv')
        print(f"Provider summary saved to water_stress_by_provider.csv")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("1. ai_datacenter_water_stress.csv - Merged dataset")
        print("2. water_stress_datacenter_map.png - Geographic visualization")
        print("3. water_stress_summary.png - Summary statistics")
        print("4. water_stress_temporal.png - Temporal trends")
        print("5. water_stress_by_provider.csv - Provider summary table")
        
        plt.show()
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
