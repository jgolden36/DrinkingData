"""
Texas Water Usage and AI Data Center Visualizations
Creates comprehensive visualizations including:
- Choropleth maps of water usage per capita
- Time series plots comparing counties with/without hyperscalers
- Scatter plots of water usage changes vs AI datacenter capacity
- Stacked bar charts of water usage categories over time
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set working directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')
STATE_WATER_DIR = os.path.join(DATA_DIR, 'Water Data', 'State Water Data')

print("="*70)
print("TEXAS WATER USAGE VISUALIZATIONS")
print("="*70)

def load_analysis_data():
    """Load the prepared analysis data and shapefile"""
    print("\n1. Loading Data...")
    print("-"*50)
    
    os.chdir(CODE_DIR)
    
    # Load main analysis data
    df = pd.read_csv('texas_water_datacenter_analysis.csv')
    print(f"Loaded {len(df)} county-year observations")
    
    # Load Texas county boundaries
    shapefile_path = os.path.join(STATE_WATER_DIR, 'Texas_County_Boundaries_4845315375211121464')
    counties_gdf = gpd.read_file(os.path.join(shapefile_path, 'County_Boundaries.shp'))
    print(f"Loaded {len(counties_gdf)} county boundaries")
    
    # Ensure CRS is appropriate for mapping
    if counties_gdf.crs.to_epsg() != 4326:
        counties_gdf = counties_gdf.to_crs('EPSG:4326')
    
    return df, counties_gdf

def define_hyperscalers(df):
    """Define hyperscaler companies and mark counties"""
    # Define hyperscaler companies (major cloud providers)
    hyperscalers = ['Amazon', 'Microsoft', 'Google', 'Meta', 'Facebook', 'Apple', 'Oracle']
    
    # Load datacenter details to identify hyperscaler counties
    dc_df = pd.read_csv('texas_datacenters_with_counties.csv')
    
    # Create hyperscaler flag
    hyperscaler_pattern = '|'.join(hyperscalers)
    dc_df['is_hyperscaler'] = dc_df['PROVIDER_NAME'].str.contains(
        hyperscaler_pattern, case=False, na=False
    )
    
    # Get counties with hyperscalers
    hyperscaler_counties = dc_df[dc_df['is_hyperscaler']]['CNTY_NM'].unique()
    
    # Add hyperscaler flag to main dataframe
    df['has_hyperscaler'] = df['CountyName'].str.upper().isin(
        [c.upper() if pd.notna(c) else '' for c in hyperscaler_counties]
    ).astype(int)
    
    print(f"Counties with hyperscaler data centers: {df.groupby('CountyName')['has_hyperscaler'].max().sum()}")
    
    return df

def create_choropleth_maps(df, counties_gdf):
    """Create choropleth maps for water usage per capita"""
    print("\n2. Creating Choropleth Maps...")
    print("-"*50)
    
    # Create figure with subplots for different maps
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    # Years to compare
    years = [2018, 2023]
    categories = ['Total', 'Power', 'Municipal']
    
    # Calculate total water usage (sum of main categories)
    water_categories = ['Municipal', 'Manufacturing', 'Mining', 'Power', 'Irrigation', 'LiveStock']
    df['Total_Water'] = df[water_categories].sum(axis=1)
    
    # Calculate per capita usage
    df['Total_Water_PerCapita'] = (df['Total_Water'] * 1000000) / df['Population']  # gallons per person
    df['Power_PerCapita'] = (df['Power'] * 1000000) / df['Population']
    df['Municipal_PerCapita'] = (df['Municipal'] * 1000000) / df['Population']
    
    plot_idx = 0
    for cat_idx, category in enumerate(categories):
        for year_idx, year in enumerate(years):
            ax = fig.add_subplot(gs[cat_idx, year_idx])
            
            # Filter data for the year
            year_data = df[df['Year'] == year].copy()
            
            # Prepare column name
            if category == 'Total':
                col_name = 'Total_Water_PerCapita'
                title_category = 'Total Water'
            else:
                col_name = f'{category}_PerCapita'
                title_category = category
            
            # Merge with shapefile
            # Clean county names for matching
            counties_gdf['CNTY_NM_UPPER'] = counties_gdf['CNTY_NM'].str.upper()
            year_data['CountyName_UPPER'] = year_data['CountyName'].str.upper()
            
            map_data = counties_gdf.merge(
                year_data[['CountyName_UPPER', col_name, 'has_ai_datacenter', 'has_datacenter_major_player']],
                left_on='CNTY_NM_UPPER',
                right_on='CountyName_UPPER',
                how='left'
            )
            
            # Create choropleth
            vmin = map_data[col_name].quantile(0.01)
            vmax = map_data[col_name].quantile(0.99)
            
            # Base map with water usage
            map_data.plot(
                column=col_name,
                ax=ax,
                legend=False,
                cmap='YlOrRd',
                vmin=vmin,
                vmax=vmax,
                edgecolor='gray',
                linewidth=0.3
            )
            
            # Overlay AI data centers with thicker borders
            ai_counties = map_data[map_data['has_ai_datacenter'] == 1]
            if len(ai_counties) > 0:
                ai_counties.boundary.plot(
                    ax=ax,
                    edgecolor='blue',
                    linewidth=2.5,
                    label='AI Data Center'
                )
            
            # Overlay major player data centers
            major_counties = map_data[
                (map_data['has_datacenter_major_player'] == 1) & 
                (map_data['has_ai_datacenter'] != 1)
            ]
            if len(major_counties) > 0:
                major_counties.boundary.plot(
                    ax=ax,
                    edgecolor='green',
                    linewidth=2,
                    linestyle='--',
                    label='Major Player DC'
                )
            
            # Title and formatting
            ax.set_title(f'{title_category} Usage Per Capita - {year}', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add legend for first map only
            if plot_idx == 0:
                legend_elements = [
                    mpatches.Rectangle((0, 0), 1, 1, 
                                      facecolor='none', 
                                      edgecolor='blue', 
                                      linewidth=2.5,
                                      label='AI Data Center County'),
                    mpatches.Rectangle((0, 0), 1, 1, 
                                      facecolor='none', 
                                      edgecolor='green', 
                                      linewidth=2,
                                      linestyle='--',
                                      label='Major Player DC County')
                ]
                ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
            
            # Add colorbar for rightmost maps
            if year_idx == 1:
                sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                                          norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
                cbar.set_label('Gallons per Capita', rotation=270, labelpad=15, fontsize=10)
            
            plot_idx += 1
    
    # Add comparison column (2023 vs 2018 change)
    for cat_idx, category in enumerate(categories):
        ax = fig.add_subplot(gs[cat_idx, 2])
        
        # Calculate change
        if category == 'Total':
            col_name = 'Total_Water_PerCapita'
        else:
            col_name = f'{category}_PerCapita'
        
        data_2018 = df[df['Year'] == 2018][['CountyName', col_name]].rename(columns={col_name: f'{col_name}_2018'})
        data_2023 = df[df['Year'] == 2023][['CountyName', col_name]].rename(columns={col_name: f'{col_name}_2023'})
        
        change_data = data_2018.merge(data_2023, on='CountyName')
        change_data['pct_change'] = ((change_data[f'{col_name}_2023'] - change_data[f'{col_name}_2018']) / 
                                     change_data[f'{col_name}_2018'] * 100)
        
        # Merge with counties for AI/DC info
        year_2023 = df[df['Year'] == 2023][['CountyName', 'has_ai_datacenter', 'has_datacenter_major_player']]
        change_data = change_data.merge(year_2023, on='CountyName')
        
        # Clean county names
        change_data['CountyName_UPPER'] = change_data['CountyName'].str.upper()
        
        # Merge with shapefile
        map_data = counties_gdf.merge(
            change_data[['CountyName_UPPER', 'pct_change', 'has_ai_datacenter', 'has_datacenter_major_player']],
            left_on='CNTY_NM_UPPER',
            right_on='CountyName_UPPER',
            how='left'
        )
        
        # Create diverging colormap centered at 0
        vmax_abs = np.abs(map_data['pct_change'].quantile([0.01, 0.99])).max()
        
        # Plot change map
        map_data.plot(
            column='pct_change',
            ax=ax,
            legend=False,
            cmap='RdBu_r',
            vmin=-vmax_abs,
            vmax=vmax_abs,
            edgecolor='gray',
            linewidth=0.3
        )
        
        # Overlay AI and major player borders
        ai_counties = map_data[map_data['has_ai_datacenter'] == 1]
        if len(ai_counties) > 0:
            ai_counties.boundary.plot(ax=ax, edgecolor='blue', linewidth=2.5)
        
        major_counties = map_data[(map_data['has_datacenter_major_player'] == 1) & 
                                 (map_data['has_ai_datacenter'] != 1)]
        if len(major_counties) > 0:
            major_counties.boundary.plot(ax=ax, edgecolor='green', linewidth=2, linestyle='--')
        
        # Title and formatting
        title_category = 'Total Water' if category == 'Total' else category
        ax.set_title(f'{title_category} Per Capita\n% Change 2018→2023', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', 
                                  norm=plt.Normalize(vmin=-vmax_abs, vmax=vmax_abs))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('% Change', rotation=270, labelpad=15, fontsize=10)
    
    plt.suptitle('Texas Water Usage Per Capita - County Level Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('texas_water_choropleth_maps.png', dpi=300, bbox_inches='tight')
    print("Saved: texas_water_choropleth_maps.png")
    plt.close()

def create_indexed_time_series(df):
    """Create indexed time series comparing counties with/without hyperscalers and AI data centers"""
    print("\n3. Creating Indexed Time Series...")
    print("-"*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate total water usage
    water_categories = ['Municipal', 'Manufacturing', 'Mining', 'Power', 'Irrigation', 'LiveStock']
    df['Total_Water'] = df[water_categories].sum(axis=1)
    df['Water_PerCapita'] = (df['Total_Water'] * 1000000) / df['Population']
    
    # 1. Hyperscaler vs Non-hyperscaler counties
    ax1 = axes[0, 0]
    
    # Group by year and hyperscaler status
    hyperscaler_trend = df.groupby(['Year', 'has_hyperscaler'])['Water_PerCapita'].mean().reset_index()
    hyperscaler_pivot = hyperscaler_trend.pivot(index='Year', columns='has_hyperscaler', values='Water_PerCapita')
    
    # Index to 2020 = 100
    base_year = 2020
    if base_year in hyperscaler_pivot.index:
        for col in hyperscaler_pivot.columns:
            base_value = hyperscaler_pivot.loc[base_year, col]
            hyperscaler_pivot[col] = (hyperscaler_pivot[col] / base_value) * 100
    
    # Plot
    hyperscaler_pivot.plot(ax=ax1, linewidth=2.5, marker='o', markersize=5)
    ax1.set_title('Water Usage Index: Hyperscaler vs Non-Hyperscaler Counties\n(2020 = 100)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Index (2020 = 100)')
    ax1.legend(['No Hyperscaler', 'Has Hyperscaler'], loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=2022, color='red', linestyle='--', alpha=0.3, label='ChatGPT Release')
    
    # 2. AI vs Non-AI datacenter counties
    ax2 = axes[0, 1]
    
    ai_trend = df.groupby(['Year', 'has_ai_datacenter'])['Water_PerCapita'].mean().reset_index()
    ai_pivot = ai_trend.pivot(index='Year', columns='has_ai_datacenter', values='Water_PerCapita')
    
    # Index to 2020 = 100
    if base_year in ai_pivot.index:
        for col in ai_pivot.columns:
            base_value = ai_pivot.loc[base_year, col]
            ai_pivot[col] = (ai_pivot[col] / base_value) * 100
    
    # Plot
    ai_pivot.plot(ax=ax2, linewidth=2.5, marker='s', markersize=5)
    ax2.set_title('Water Usage Index: AI vs Non-AI Datacenter Counties\n(2020 = 100)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Index (2020 = 100)')
    ax2.legend(['No AI Datacenter', 'Has AI Datacenter'], loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=2022, color='red', linestyle='--', alpha=0.3, label='ChatGPT Release')
    
    # 3. Absolute water usage comparison - Hyperscalers
    ax3 = axes[1, 0]
    
    hyperscaler_abs = df.groupby(['Year', 'has_hyperscaler'])['Water_PerCapita'].mean().reset_index()
    hyperscaler_abs_pivot = hyperscaler_abs.pivot(index='Year', columns='has_hyperscaler', values='Water_PerCapita')
    
    hyperscaler_abs_pivot.plot(ax=ax3, linewidth=2.5, marker='o', markersize=5, color=['#2E7D32', '#1976D2'])
    ax3.set_title('Absolute Water Usage Per Capita: Hyperscaler Counties', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Gallons per Capita')
    ax3.legend(['No Hyperscaler', 'Has Hyperscaler'], loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Add shaded region for post-ChatGPT
    ax3.axvspan(2022, 2023, alpha=0.1, color='red')
    
    # 4. Absolute water usage comparison - AI Datacenters
    ax4 = axes[1, 1]
    
    ai_abs = df.groupby(['Year', 'has_ai_datacenter'])['Water_PerCapita'].mean().reset_index()
    ai_abs_pivot = ai_abs.pivot(index='Year', columns='has_ai_datacenter', values='Water_PerCapita')
    
    ai_abs_pivot.plot(ax=ax4, linewidth=2.5, marker='s', markersize=5, color=['#E65100', '#7B1FA2'])
    ax4.set_title('Absolute Water Usage Per Capita: AI Datacenter Counties', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Gallons per Capita')
    ax4.legend(['No AI Datacenter', 'Has AI Datacenter'], loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Add shaded region for post-ChatGPT
    ax4.axvspan(2022, 2023, alpha=0.1, color='red')
    
    plt.suptitle('Texas Water Usage Trends: Impact of Hyperscaler and AI Data Centers', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('texas_water_indexed_timeseries.png', dpi=300, bbox_inches='tight')
    print("Saved: texas_water_indexed_timeseries.png")
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    
    # Calculate growth rates
    if 2023 in hyperscaler_pivot.index and 2018 in hyperscaler_pivot.index:
        hyper_growth_no = ((hyperscaler_pivot.loc[2023, 0] / hyperscaler_pivot.loc[2018, 0]) - 1) * 100
        hyper_growth_yes = ((hyperscaler_pivot.loc[2023, 1] / hyperscaler_pivot.loc[2018, 1]) - 1) * 100
        print(f"Hyperscaler counties growth 2018-2023: {hyper_growth_yes:.1f}%")
        print(f"Non-hyperscaler counties growth 2018-2023: {hyper_growth_no:.1f}%")
        
    if 2023 in ai_pivot.index and 2018 in ai_pivot.index:
        ai_growth_no = ((ai_pivot.loc[2023, 0] / ai_pivot.loc[2018, 0]) - 1) * 100
        ai_growth_yes = ((ai_pivot.loc[2023, 1] / ai_pivot.loc[2018, 1]) - 1) * 100
        print(f"AI datacenter counties growth 2018-2023: {ai_growth_yes:.1f}%")
        print(f"Non-AI datacenter counties growth 2018-2023: {ai_growth_no:.1f}%")

def create_scatter_plot(df):
    """Create scatter plot of water usage change vs AI datacenter capacity"""
    print("\n4. Creating Scatter Plot...")
    print("-"*50)
    
    # Calculate water usage change from 2018 to 2023
    water_categories = ['Municipal', 'Manufacturing', 'Mining', 'Power', 'Irrigation', 'LiveStock']
    df['Total_Water'] = df[water_categories].sum(axis=1)
    df['Water_PerCapita'] = (df['Total_Water'] * 1000000) / df['Population']
    
    # Get 2018 and 2023 data
    data_2018 = df[df['Year'] == 2018][['CountyName', 'Water_PerCapita']].rename(
        columns={'Water_PerCapita': 'Water_PerCapita_2018'}
    )
    data_2023 = df[df['Year'] == 2023][['CountyName', 'Water_PerCapita', 'ai_datacenter_capacity_mw', 
                                        'has_ai_datacenter', 'has_datacenter_major_player']].rename(
        columns={'Water_PerCapita': 'Water_PerCapita_2023'}
    )
    
    # Merge and calculate percentage change
    scatter_data = data_2018.merge(data_2023, on='CountyName')
    scatter_data['pct_change'] = ((scatter_data['Water_PerCapita_2023'] - 
                                   scatter_data['Water_PerCapita_2018']) / 
                                  scatter_data['Water_PerCapita_2018'] * 100)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Scatter plot: % change vs AI datacenter capacity
    ax1 = axes[0]
    
    # Separate AI and non-AI counties
    ai_counties = scatter_data[scatter_data['has_ai_datacenter'] == 1]
    non_ai_counties = scatter_data[scatter_data['has_ai_datacenter'] == 0]
    
    # Plot non-AI counties as background
    ax1.scatter(non_ai_counties['ai_datacenter_capacity_mw'], 
               non_ai_counties['pct_change'],
               alpha=0.3, s=30, color='gray', label='No AI Datacenter')
    
    # Plot AI counties with size based on capacity
    scatter = ax1.scatter(ai_counties['ai_datacenter_capacity_mw'], 
                         ai_counties['pct_change'],
                         s=ai_counties['ai_datacenter_capacity_mw']/10 + 50,
                         alpha=0.6, 
                         c=ai_counties['ai_datacenter_capacity_mw'],
                         cmap='Reds',
                         edgecolors='black',
                         linewidth=1,
                         label='Has AI Datacenter')
    
    # Add trend line for AI counties
    if len(ai_counties) > 1:
        z = np.polyfit(ai_counties['ai_datacenter_capacity_mw'], ai_counties['pct_change'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(ai_counties['ai_datacenter_capacity_mw'].min(), 
                             ai_counties['ai_datacenter_capacity_mw'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax1.set_xlabel('AI Datacenter Capacity (MW)', fontsize=12)
    ax1.set_ylabel('% Change in Water Usage Per Capita (2018-2023)', fontsize=12)
    ax1.set_title('Water Usage Change vs AI Datacenter Capacity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, fraction=0.03, pad=0.04)
    cbar.set_label('AI Capacity (MW)', rotation=270, labelpad=15)
    
    # 2. Box plot comparison
    ax2 = axes[1]
    
    # Prepare data for box plot
    box_data = []
    labels = []
    
    # No datacenter
    no_dc = scatter_data[(scatter_data['has_ai_datacenter'] == 0) & 
                        (scatter_data['has_datacenter_major_player'] == 0)]
    box_data.append(no_dc['pct_change'].dropna())
    labels.append(f'No Datacenter\n(n={len(no_dc)})')
    
    # Major player but no AI
    major_only = scatter_data[(scatter_data['has_ai_datacenter'] == 0) & 
                             (scatter_data['has_datacenter_major_player'] == 1)]
    box_data.append(major_only['pct_change'].dropna())
    labels.append(f'Major Player Only\n(n={len(major_only)})')
    
    # Has AI datacenter
    has_ai = scatter_data[scatter_data['has_ai_datacenter'] == 1]
    box_data.append(has_ai['pct_change'].dropna())
    labels.append(f'AI Datacenter\n(n={len(has_ai)})')
    
    # Create box plot
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # Color the boxes
    colors = ['lightgray', 'lightblue', 'salmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('% Change in Water Usage Per Capita (2018-2023)', fontsize=12)
    ax2.set_title('Water Usage Change by Datacenter Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add mean values as text
    for i, data in enumerate(box_data):
        mean_val = data.mean()
        ax2.text(i+1, ax2.get_ylim()[1]*0.95, f'μ={mean_val:.1f}%', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Texas County Water Usage Changes and AI Datacenter Capacity (2018-2023)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('texas_water_scatter_plot.png', dpi=300, bbox_inches='tight')
    print("Saved: texas_water_scatter_plot.png")
    plt.close()
    
    # Print correlation statistics
    print("\nCorrelation Analysis:")
    print("-" * 50)
    
    # For AI counties only
    if len(ai_counties) > 0:
        corr = ai_counties[['ai_datacenter_capacity_mw', 'pct_change']].corr().iloc[0, 1]
        print(f"Correlation (AI counties): {corr:.3f}")
        
        # Top 5 counties by capacity
        top_ai = ai_counties.nlargest(5, 'ai_datacenter_capacity_mw')[
            ['CountyName', 'ai_datacenter_capacity_mw', 'pct_change']
        ]
        print("\nTop 5 AI Counties by Capacity:")
        for _, row in top_ai.iterrows():
            print(f"  {row['CountyName']}: {row['ai_datacenter_capacity_mw']:.0f} MW, "
                 f"{row['pct_change']:.1f}% change")

def create_stacked_bar_charts(df):
    """Create stacked bar charts for water usage categories over time"""
    print("\n5. Creating Stacked Bar Charts...")
    print("-"*50)
    
    # Water usage categories to plot
    water_categories = ['Municipal', 'Manufacturing', 'Mining', 'Power', 'Irrigation', 'LiveStock']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Define colors for each category
    colors = {
        'Municipal': '#1976D2',
        'Manufacturing': '#FFA726',
        'Mining': '#66BB6A',
        'Power': '#EF5350',
        'Irrigation': '#42A5F5',
        'LiveStock': '#AB47BC'
    }
    
    # 1. Counties WITH AI datacenters
    ax1 = axes[0]
    
    # Filter for counties with AI datacenters
    ai_counties_data = df[df['has_ai_datacenter'] == 1]
    
    # Group by year and sum water usage
    ai_yearly = ai_counties_data.groupby('Year')[water_categories].sum()
    
    # Create stacked bar chart
    bottom = np.zeros(len(ai_yearly))
    for category in water_categories:
        ax1.bar(ai_yearly.index, ai_yearly[category], 
               bottom=bottom, label=category, 
               color=colors[category], alpha=0.8, width=0.8)
        bottom += ai_yearly[category]
    
    ax1.set_title(f'Water Usage in Counties WITH AI Datacenters ({len(ai_counties_data.CountyName.unique())} counties)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Water Usage (Million Gallons)', fontsize=12)
    ax1.legend(loc='upper left', ncol=len(water_categories))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add vertical line for ChatGPT release
    ax1.axvline(x=2022, color='red', linestyle='--', alpha=0.5, linewidth=2, label='ChatGPT Release')
    
    # Add text annotation for total
    for year in [2018, 2023]:
        if year in ai_yearly.index:
            total = ai_yearly.loc[year].sum()
            ax1.text(year, total + total*0.01, f'{total:,.0f}', 
                    ha='center', fontsize=10, fontweight='bold')
    
    # 2. Counties WITHOUT AI datacenters
    ax2 = axes[1]
    
    # Filter for counties without AI datacenters
    non_ai_counties_data = df[df['has_ai_datacenter'] == 0]
    
    # Group by year and sum water usage
    non_ai_yearly = non_ai_counties_data.groupby('Year')[water_categories].sum()
    
    # Create stacked bar chart
    bottom = np.zeros(len(non_ai_yearly))
    for category in water_categories:
        ax2.bar(non_ai_yearly.index, non_ai_yearly[category], 
               bottom=bottom, label=category, 
               color=colors[category], alpha=0.8, width=0.8)
        bottom += non_ai_yearly[category]
    
    ax2.set_title(f'Water Usage in Counties WITHOUT AI Datacenters ({len(non_ai_counties_data.CountyName.unique())} counties)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Water Usage (Million Gallons)', fontsize=12)
    ax2.legend(loc='upper left', ncol=len(water_categories))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add vertical line for ChatGPT release
    ax2.axvline(x=2022, color='red', linestyle='--', alpha=0.5, linewidth=2, label='ChatGPT Release')
    
    # Add text annotation for total
    for year in [2018, 2023]:
        if year in non_ai_yearly.index:
            total = non_ai_yearly.loc[year].sum()
            ax2.text(year, total + total*0.01, f'{total:,.0f}', 
                    ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Texas Water Usage by Category: AI vs Non-AI Datacenter Counties', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('texas_water_stacked_bars.png', dpi=300, bbox_inches='tight')
    print("Saved: texas_water_stacked_bars.png")
    plt.close()
    
    # Create percentage composition chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate percentage composition for 2023
    year_2023 = 2023
    
    # AI counties composition
    ax1 = axes[0]
    if year_2023 in ai_yearly.index:
        ai_2023 = ai_yearly.loc[year_2023]
        ai_pct = (ai_2023 / ai_2023.sum() * 100)
        
        wedges, texts, autotexts = ax1.pie(ai_pct, labels=ai_pct.index, 
                                           colors=[colors[cat] for cat in ai_pct.index],
                                           autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'2023 Water Usage Composition\nCounties WITH AI Datacenters', 
                     fontsize=14, fontweight='bold')
    
    # Non-AI counties composition
    ax2 = axes[1]
    if year_2023 in non_ai_yearly.index:
        non_ai_2023 = non_ai_yearly.loc[year_2023]
        non_ai_pct = (non_ai_2023 / non_ai_2023.sum() * 100)
        
        wedges, texts, autotexts = ax2.pie(non_ai_pct, labels=non_ai_pct.index,
                                          colors=[colors[cat] for cat in non_ai_pct.index],
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'2023 Water Usage Composition\nCounties WITHOUT AI Datacenters', 
                     fontsize=14, fontweight='bold')
    
    plt.suptitle('Water Usage Category Distribution in 2023', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('texas_water_composition_2023.png', dpi=300, bbox_inches='tight')
    print("Saved: texas_water_composition_2023.png")
    plt.close()
    
    # Print summary statistics
    print("\nWater Usage Summary (2023):")
    print("-" * 50)
    
    if year_2023 in ai_yearly.index:
        ai_total_2023 = ai_yearly.loc[year_2023].sum()
        print(f"AI Datacenter Counties Total: {ai_total_2023:,.0f} Million Gallons")
        print("  Category breakdown:")
        for cat in water_categories:
            val = ai_yearly.loc[year_2023, cat]
            pct = val / ai_total_2023 * 100
            print(f"    {cat}: {val:,.0f} ({pct:.1f}%)")
    
    if year_2023 in non_ai_yearly.index:
        non_ai_total_2023 = non_ai_yearly.loc[year_2023].sum()
        print(f"\nNon-AI Datacenter Counties Total: {non_ai_total_2023:,.0f} Million Gallons")
        print("  Category breakdown:")
        for cat in water_categories:
            val = non_ai_yearly.loc[year_2023, cat]
            pct = val / non_ai_total_2023 * 100
            print(f"    {cat}: {val:,.0f} ({pct:.1f}%)")
    
    # Calculate growth rates
    if 2018 in ai_yearly.index and 2023 in ai_yearly.index:
        print("\nGrowth Rates 2018-2023 (AI Datacenter Counties):")
        for cat in water_categories:
            growth = ((ai_yearly.loc[2023, cat] / ai_yearly.loc[2018, cat]) - 1) * 100
            print(f"  {cat}: {growth:.1f}%")

def main():
    """Main execution function"""
    try:
        # Load data
        df, counties_gdf = load_analysis_data()
        
        # Define hyperscaler companies
        df = define_hyperscalers(df)
        
        # Create visualizations
        create_choropleth_maps(df, counties_gdf)
        create_indexed_time_series(df)
        create_scatter_plot(df)
        create_stacked_bar_charts(df)
        
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("1. texas_water_choropleth_maps.png - Per capita usage maps")
        print("2. texas_water_indexed_timeseries.png - Indexed trends comparison")
        print("3. texas_water_scatter_plot.png - Capacity vs usage change analysis")
        print("4. texas_water_stacked_bars.png - Category breakdown over time")
        print("5. texas_water_composition_2023.png - Usage composition pie charts")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
