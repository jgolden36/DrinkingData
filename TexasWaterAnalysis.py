"""
Texas Water Usage and AI Data Center Analysis
This script combines Texas state water data with data center locations
to analyze the impact of AI data centers on water usage patterns.

Focuses on county-level analysis from 2000 onwards with special attention
to the impact of ChatGPT's release in 2022.
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point
import glob
import warnings
warnings.filterwarnings('ignore')

# Set working directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')
STATE_WATER_DIR = os.path.join(DATA_DIR, 'Water Data', 'State Water Data')

print("="*70)
print("TEXAS WATER USAGE AND AI DATA CENTER ANALYSIS")
print("="*70)

def load_and_combine_texas_water_data():
    """
    Load and combine all Texas water usage CSV files
    """
    print("\n1. Loading Texas Water Usage Data...")
    print("-"*50)
    
    # Find all Texas water data files
    water_files = glob.glob(os.path.join(STATE_WATER_DIR, 'SumFinal_CountyReportWithReuse*.csv'))
    print(f"Found {len(water_files)} water data files")
    
    # Load and combine all files
    dfs = []
    for file in water_files:
        try:
            df = pd.read_csv(file, thousands=',')
            dfs.append(df)
            print(f"  Loaded: {os.path.basename(file)} ({len(df)} records)")
        except Exception as e:
            print(f"  Error loading {os.path.basename(file)}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates based on County and Year
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['CountyName', 'Year'])
    
    print(f"\nCombined data: {initial_count} total records")
    print(f"After removing duplicates: {len(combined_df)} unique county-year records")
    print(f"Years covered: {combined_df['Year'].min()} - {combined_df['Year'].max()}")
    print(f"Counties: {combined_df['CountyName'].nunique()}")
    
    # Display available water usage columns
    water_columns = [col for col in combined_df.columns if col not in ['CountyName', 'Year', 'Population']]
    print(f"\nWater usage categories found: {len(water_columns)}")
    print("Sample categories:", water_columns[:10])
    
    return combined_df

def load_texas_county_boundaries():
    """
    Load Texas county boundaries shapefile
    """
    print("\n2. Loading Texas County Boundaries...")
    print("-"*50)
    
    # Find the shapefile
    shapefile_path = os.path.join(STATE_WATER_DIR, 'Texas_County_Boundaries_4845315375211121464')
    
    # Check for .shp file in the directory
    shp_files = glob.glob(os.path.join(shapefile_path, '*.shp'))
    
    if shp_files:
        # Load the shapefile
        counties_gdf = gpd.read_file(shp_files[0])
        print(f"Loaded county boundaries: {len(counties_gdf)} counties")
        print(f"CRS: {counties_gdf.crs}")
        
        # Check column names to identify county name field
        print("\nShapefile columns:", list(counties_gdf.columns))
        
        # Try to identify county name column
        name_columns = [col for col in counties_gdf.columns if 'NAME' in col.upper() or 'COUNTY' in col.upper()]
        if name_columns:
            print(f"Potential county name columns: {name_columns}")
            
        return counties_gdf
    else:
        print(f"Warning: No shapefile found in {shapefile_path}")
        return None

def load_and_filter_texas_datacenters():
    """
    Load data center inventory and filter for Texas
    """
    print("\n3. Loading Data Center Inventory...")
    print("-"*50)
    
    os.chdir(DATA_DIR)
    
    # Load the latest data center inventory
    try:
        # Try the newest file first
        datacenter_df = pd.read_csv('data_center_inventory_20251217.csv', thousands=',')
    except:
        # Fallback to older file if newer doesn't exist
        datacenter_df = pd.read_csv('data_center_inventory_20250523.csv', thousands=',')
    
    print(f"Total data centers loaded: {len(datacenter_df)}")
    
    # Filter for Texas (using state name or coordinates)
    # First try using STATE_NAME if available
    if 'STATE_NAME' in datacenter_df.columns:
        texas_mask = datacenter_df['STATE_NAME'].str.upper() == 'TEXAS'
        texas_datacenters = datacenter_df[texas_mask].copy()
    else:
        # Use coordinate boundaries for Texas (approximate)
        texas_mask = (
            (datacenter_df['LOCATION_LONGITUDE'] >= -107) & 
            (datacenter_df['LOCATION_LONGITUDE'] <= -93) &
            (datacenter_df['LOCATION_LATITUDE'] >= 25.5) & 
            (datacenter_df['LOCATION_LATITUDE'] <= 36.5)
        )
        texas_datacenters = datacenter_df[texas_mask].copy()
    
    print(f"Texas data centers: {len(texas_datacenters)}")
    
    # Convert to GeoDataFrame
    texas_datacenters['geometry'] = texas_datacenters.apply(
        lambda row: Point(row['LOCATION_LONGITUDE'], row['LOCATION_LATITUDE']), axis=1
    )
    texas_datacenters_gdf = gpd.GeoDataFrame(texas_datacenters, geometry='geometry', crs='EPSG:4326')
    
    # Identify AI data centers (if flag exists)
    if 'FLG_AI_FACILITY' in texas_datacenters.columns:
        ai_count = (texas_datacenters['FLG_AI_FACILITY'] == 'Y').sum()
        print(f"AI data centers in Texas: {ai_count}")
    else:
        print("AI facility flag not found, will use provider-based identification")
    
    # Identify major players
    major_players = ['Google', 'Microsoft', 'Amazon', 'Meta', 'Facebook', 'Apple', 'Oracle', 'IBM']
    major_player_pattern = '|'.join(major_players)
    texas_datacenters['is_major_player'] = texas_datacenters['PROVIDER_NAME'].str.contains(
        major_player_pattern, case=False, na=False
    )
    print(f"Data centers by major players: {texas_datacenters['is_major_player'].sum()}")
    
    # Make sure the column is in the GeoDataFrame
    texas_datacenters_gdf['is_major_player'] = texas_datacenters['is_major_player']
    
    # Debug: Check columns before returning
    print(f"Columns in GeoDataFrame: {list(texas_datacenters_gdf.columns)[:10]}...")
    
    return texas_datacenters_gdf

def map_datacenters_to_counties(datacenters_gdf, counties_gdf):
    """
    Perform spatial join to map data centers to Texas counties
    """
    print("\n4. Mapping Data Centers to Counties...")
    print("-"*50)
    
    # Store columns from original datacenters before join
    original_columns = list(datacenters_gdf.columns)
    
    # Ensure both GeoDataFrames are in the same CRS
    if counties_gdf.crs != datacenters_gdf.crs:
        print(f"Converting CRS from {datacenters_gdf.crs} to {counties_gdf.crs}")
        datacenters_gdf = datacenters_gdf.to_crs(counties_gdf.crs)
    
    # Perform spatial join
    datacenters_with_counties = gpd.sjoin(
        datacenters_gdf, 
        counties_gdf, 
        how='left', 
        predicate='within'
    )
    
    print(f"Spatial join complete: {len(datacenters_with_counties)} data centers mapped")
    
    # Check how many were successfully mapped
    mapped_count = datacenters_with_counties['index_right'].notna().sum()
    print(f"Successfully mapped to counties: {mapped_count}/{len(datacenters_with_counties)}")
    
    # Verify important columns are preserved
    if 'is_major_player' in original_columns and 'is_major_player' not in datacenters_with_counties.columns:
        print("Warning: 'is_major_player' column was lost during spatial join")
    
    return datacenters_with_counties

def create_county_datacenter_features(datacenters_with_counties, counties_gdf, water_df):
    """
    Create county-level features for data center presence and capacity
    """
    print("\n5. Creating County-Level Data Center Features...")
    print("-"*50)
    
    # Identify the county name column in the shapefile - look for CNTY_NM or similar
    county_name_col = None
    for col in counties_gdf.columns:
        if 'CNTY_NM' in col or 'CNTY_NAME' in col.upper() or ('COUNTY' in col.upper() and 'NAME' in col.upper()):
            county_name_col = col
            break
    
    # Fallback to other name columns
    if not county_name_col:
        for col in counties_gdf.columns:
            if 'NAME' in col.upper() or 'NM' in col.upper():
                county_name_col = col
                break
    
    if not county_name_col:
        print("Warning: Could not identify county name column in shapefile")
        county_name_col = 'CNTY_NM'  # Use CNTY_NM based on what we saw in the columns
    
    print(f"Using '{county_name_col}' as county name column")
    
    # Create a mapping dictionary from index to county name
    county_mapping = counties_gdf[county_name_col].to_dict()
    
    # Get county name from spatial join - handle NaN values
    datacenters_with_counties['county_from_shapefile'] = datacenters_with_counties['index_right'].map(county_mapping)
    
    # Convert to string and handle NaN
    datacenters_with_counties['county_from_shapefile'] = datacenters_with_counties['county_from_shapefile'].fillna('').astype(str)
    
    # Create aggregations by county
    county_features = pd.DataFrame()
    
    # Get unique counties from water data
    unique_counties = water_df['CountyName'].unique()
    county_features['CountyName'] = unique_counties
    
    # Initialize features
    county_features['has_datacenter_major_player'] = 0
    county_features['has_ai_datacenter'] = 0
    county_features['ai_datacenter_capacity_mw'] = 0.0
    county_features['total_datacenter_count'] = 0
    county_features['total_datacenter_capacity_mw'] = 0.0
    
    # Aggregate data center features by county
    for county in unique_counties:
        # Try to match county names (handle case and 'County' suffix)
        county_clean = str(county).upper().replace(' COUNTY', '').strip()
        
        # Find data centers in this county - ensure column is string type
        county_from_sf = datacenters_with_counties['county_from_shapefile'].astype(str)
        county_mask = county_from_sf.str.upper().str.replace(' COUNTY', '').str.strip() == county_clean
        county_datacenters = datacenters_with_counties[county_mask]
        
        if len(county_datacenters) > 0:
            idx = county_features['CountyName'] == county
            
            # Has major player data center
            county_features.loc[idx, 'has_datacenter_major_player'] = int(
                county_datacenters['is_major_player'].any()
            )
            
            # Has AI data center
            if 'FLG_AI_FACILITY' in county_datacenters.columns:
                county_features.loc[idx, 'has_ai_datacenter'] = int(
                    (county_datacenters['FLG_AI_FACILITY'] == 'Y').any()
                )
                # AI data center capacity
                ai_datacenters = county_datacenters[county_datacenters['FLG_AI_FACILITY'] == 'Y']
                county_features.loc[idx, 'ai_datacenter_capacity_mw'] = ai_datacenters['SELECTED_POWER_CAPACITY_MW'].sum()
            
            # Total counts and capacity
            county_features.loc[idx, 'total_datacenter_count'] = len(county_datacenters)
            county_features.loc[idx, 'total_datacenter_capacity_mw'] = county_datacenters['SELECTED_POWER_CAPACITY_MW'].sum()
    
    print(f"\nCounty-level features created:")
    print(f"  Counties with major player data centers: {county_features['has_datacenter_major_player'].sum()}")
    print(f"  Counties with AI data centers: {county_features['has_ai_datacenter'].sum()}")
    print(f"  Total AI capacity (MW): {county_features['ai_datacenter_capacity_mw'].sum():.1f}")
    print(f"  Counties with any data centers: {(county_features['total_datacenter_count'] > 0).sum()}")
    
    return county_features

def merge_water_and_datacenter_data(water_df, county_features):
    """
    Merge water usage data with data center features and add ChatGPT dummy
    """
    print("\n6. Merging Water Data with Data Center Features...")
    print("-"*50)
    
    # Merge water data with county features
    merged_df = water_df.merge(
        county_features,
        on='CountyName',
        how='left'
    )
    
    # Fill NaN values with 0 for data center features
    datacenter_cols = ['has_datacenter_major_player', 'has_ai_datacenter', 
                      'ai_datacenter_capacity_mw', 'total_datacenter_count', 
                      'total_datacenter_capacity_mw']
    merged_df[datacenter_cols] = merged_df[datacenter_cols].fillna(0)
    
    # Create interaction term: AI datacenter × Capacity MW
    merged_df['ai_datacenter_x_capacity'] = merged_df['has_ai_datacenter'] * merged_df['ai_datacenter_capacity_mw']
    
    # Add ChatGPT release dummy (1 for 2022 and later)
    merged_df['post_chatgpt'] = (merged_df['Year'] >= 2022).astype(int)
    
    print(f"Merged dataset: {len(merged_df)} county-year observations")
    print(f"Years: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
    print(f"Counties: {merged_df['CountyName'].nunique()}")
    
    # Summary statistics
    print("\nData Center Presence Over Time:")
    yearly_summary = merged_df.groupby('Year').agg({
        'has_datacenter_major_player': 'sum',
        'has_ai_datacenter': 'sum',
        'ai_datacenter_capacity_mw': 'sum',
        'post_chatgpt': 'mean'
    })
    print(yearly_summary.tail(10))
    
    return merged_df

def save_analysis_data(merged_df, datacenters_with_counties):
    """
    Save the processed data for analysis
    """
    print("\n7. Saving Analysis Files...")
    print("-"*50)
    
    os.chdir(CODE_DIR)
    
    # Save merged water and data center data
    output_file = 'texas_water_datacenter_analysis.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"Main analysis file saved: {output_file}")
    
    # Save data center details
    dc_output = 'texas_datacenters_with_counties.csv'
    dc_save = datacenters_with_counties.drop(columns=['geometry'])
    dc_save.to_csv(dc_output, index=False)
    print(f"Data center details saved: {dc_output}")
    
    # Create summary statistics file
    summary_stats = {
        'Total County-Years': len(merged_df),
        'Years Covered': f"{merged_df['Year'].min()}-{merged_df['Year'].max()}",
        'Number of Counties': merged_df['CountyName'].nunique(),
        'Counties with Major Player DCs': merged_df.groupby('CountyName')['has_datacenter_major_player'].max().sum(),
        'Counties with AI DCs': merged_df.groupby('CountyName')['has_ai_datacenter'].max().sum(),
        'Total AI DC Capacity (MW)': merged_df.groupby('CountyName')['ai_datacenter_capacity_mw'].max().sum(),
        'Observations Post-ChatGPT': merged_df['post_chatgpt'].sum(),
        'Water Usage Variables': len([col for col in merged_df.columns if col not in 
                                     ['CountyName', 'Year', 'Population'] + 
                                     ['has_datacenter_major_player', 'has_ai_datacenter', 
                                      'ai_datacenter_capacity_mw', 'total_datacenter_count',
                                      'total_datacenter_capacity_mw', 'ai_datacenter_x_capacity', 
                                      'post_chatgpt']])
    }
    
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
    summary_df.to_csv('texas_analysis_summary.csv', index=False)
    print(f"Summary statistics saved: texas_analysis_summary.csv")
    
    return output_file

def main():
    """
    Main execution function
    """
    try:
        # Load and combine Texas water data
        water_df = load_and_combine_texas_water_data()
        
        # Load county boundaries
        counties_gdf = load_texas_county_boundaries()
        
        if counties_gdf is None:
            print("Warning: Could not load county boundaries shapefile")
            print("Analysis will continue without spatial join")
            return
        
        # Load and filter data centers
        texas_datacenters_gdf = load_and_filter_texas_datacenters()
        
        # Map data centers to counties
        datacenters_with_counties = map_datacenters_to_counties(texas_datacenters_gdf, counties_gdf)
        
        # Create county-level features
        county_features = create_county_datacenter_features(
            datacenters_with_counties, 
            counties_gdf, 
            water_df
        )
        
        # Merge water data with data center features
        merged_df = merge_water_and_datacenter_data(water_df, county_features)
        
        # Save analysis files
        output_file = save_analysis_data(merged_df, datacenters_with_counties)
        
        print("\n" + "="*70)
        print("ANALYSIS PREPARATION COMPLETE!")
        print("="*70)
        print(f"\nMain analysis file: {output_file}")
        print("Ready for difference-in-differences and panel data analysis")
        print("\nKey variables created:")
        print("  - has_datacenter_major_player: Binary indicator for major tech companies")
        print("  - has_ai_datacenter: Binary indicator for AI facilities")
        print("  - ai_datacenter_x_capacity: Interaction of AI dummy and capacity")
        print("  - post_chatgpt: Binary indicator for years >= 2022")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Script starting...")
    import sys
    sys.stdout.flush()
    main()
    print("Script finished")
