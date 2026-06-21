"""
AI Data Center Water Stress Analysis Charts
Detailed analysis and visualizations of AI facilities and water risk
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("AI DATA CENTER WATER STRESS - DETAILED ANALYSIS")
print("="*70)

# Set directories
BASE_DIR = r'C:\Users\bnhas\OneDrive\Desktop\Classes\Work Dissertation Phase\Data Center Environmental'
DATA_DIR = os.path.join(BASE_DIR, 'Data')
CODE_DIR = os.path.join(BASE_DIR, 'Code')

os.chdir(DATA_DIR)

print("\n1. Loading Data...")
print("-"*50)

# Load the new data center inventory with AI flag
datacenter_df = pd.read_csv('data_center_inventory_20251217.csv', thousands=',')

# Filter for AI facilities
ai_mask = datacenter_df['FLG_AI_FACILITY'] == 'Y'
ai_datacenters = datacenter_df[ai_mask].copy()

# Filter for US data centers
us_mask = (
    (ai_datacenters.LOCATION_LONGITUDE >= -125) & 
    (ai_datacenters.LOCATION_LONGITUDE <= -66) &
    (ai_datacenters.LOCATION_LATITUDE >= 24) & 
    (ai_datacenters.LOCATION_LATITUDE <= 50)
)
us_ai_datacenters = ai_datacenters[us_mask].copy()

print(f"Loaded {len(ai_datacenters):,} AI data centers")
print(f"US AI data centers: {len(us_ai_datacenters):,}")

# Define high water stress zones
high_water_stress_zones = {
    'California Central Valley': {
        'bbox': [-122.5, 35.5, -119, 40],
        'level': 4.5,
        'label': 'Extremely High'
    },
    'Southern California': {
        'bbox': [-120, 32.5, -115, 34.5],
        'level': 4.3,
        'label': 'Extremely High'
    },
    'Arizona - Phoenix Area': {
        'bbox': [-113, 32.5, -111, 34],
        'level': 4.4,
        'label': 'Extremely High'
    },
    'Nevada - Las Vegas': {
        'bbox': [-116, 35.5, -114, 36.5],
        'level': 4.5,
        'label': 'Extremely High'
    },
    'Texas Panhandle': {
        'bbox': [-103, 34, -100, 37],
        'level': 3.8,
        'label': 'High'
    },
    'New Mexico - Rio Grande': {
        'bbox': [-107, 32, -105, 36],
        'level': 3.7,
        'label': 'High'
    },
    'Colorado - Front Range': {
        'bbox': [-106, 38, -104, 41],
        'level': 3.5,
        'label': 'High'
    },
    'Kansas - Ogallala': {
        'bbox': [-102, 37, -99, 39],
        'level': 3.6,
        'label': 'High'
    },
    'Utah - Great Salt Lake': {
        'bbox': [-113, 40, -111, 42],
        'level': 3.9,
        'label': 'High'
    }
}

print("\n2. Analyzing Water Stress Exposure...")
print("-"*50)

# Mark data centers in high water stress zones
us_ai_datacenters['water_stress_zone'] = 'Low/None'
us_ai_datacenters['water_stress_level'] = 'Low'

for idx, row in us_ai_datacenters.iterrows():
    for zone_name, zone_info in high_water_stress_zones.items():
        x_min, y_min, x_max, y_max = zone_info['bbox']
        if x_min <= row.LOCATION_LONGITUDE <= x_max and y_min <= row.LOCATION_LATITUDE <= y_max:
            us_ai_datacenters.at[idx, 'water_stress_zone'] = zone_name
            us_ai_datacenters.at[idx, 'water_stress_level'] = zone_info['label']
            break

# Calculate statistics
total_at_risk = len(us_ai_datacenters[us_ai_datacenters['water_stress_level'] != 'Low'])
print(f"AI data centers in high water stress zones: {total_at_risk}")
print(f"Percentage at risk: {total_at_risk/len(us_ai_datacenters)*100:.1f}%")

os.chdir(CODE_DIR)

print("\n3. Creating Analysis Charts...")
print("-"*50)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Water Stress Distribution Pie Chart
ax1 = plt.subplot(3, 3, 1)
stress_counts = us_ai_datacenters['water_stress_level'].value_counts()
colors = {'Extremely High': '#8B0000', 'High': '#CD5C5C', 'Low': '#90EE90'}
wedges, texts, autotexts = ax1.pie(
    stress_counts.values, 
    labels=stress_counts.index,
    colors=[colors.get(x, '#808080') for x in stress_counts.index],
    autopct='%1.1f%%',
    startangle=90
)
ax1.set_title('AI Data Centers by Water Stress Level', fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 2. Top Providers Bar Chart
ax2 = plt.subplot(3, 3, 2)
top_providers = us_ai_datacenters['PROVIDER_NAME'].value_counts().head(10)
bars = ax2.barh(range(len(top_providers)), top_providers.values, color='steelblue')
ax2.set_yticks(range(len(top_providers)))
ax2.set_yticklabels(top_providers.index, fontsize=9)
ax2.set_xlabel('Number of AI Data Centers')
ax2.set_title('Top 10 AI Data Center Providers (US)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_providers.values)):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)

# 3. Development Stage Distribution
ax3 = plt.subplot(3, 3, 3)
stage_counts = us_ai_datacenters['DATA_CENTER_STAGE'].value_counts()
stage_colors = {
    'Active': '#2E7D32',
    'Construction': '#FFA726',
    'Announcement': '#1976D2',
    'Planning': '#7B1FA2',
    'Planned': '#7B1FA2',
    'Not Approved/Withdrawn': '#616161',
    'Delayed': '#D32F2F'
}
bars = ax3.bar(
    range(len(stage_counts)), 
    stage_counts.values,
    color=[stage_colors.get(x, '#424242') for x in stage_counts.index]
)
ax3.set_xticks(range(len(stage_counts)))
ax3.set_xticklabels(stage_counts.index, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Number of Facilities')
ax3.set_title('AI Data Centers by Development Stage', fontsize=12, fontweight='bold')
# Add value labels
for bar, val in zip(bars, stage_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 5, str(val), ha='center', fontsize=9)

# 4. Power Capacity Analysis
ax4 = plt.subplot(3, 3, 4)
# Group by water stress level and calculate total power
power_by_stress = us_ai_datacenters.groupby('water_stress_level')['SELECTED_POWER_CAPACITY_MW'].sum() / 1000  # Convert to GW
bars = ax4.bar(
    range(len(power_by_stress)), 
    power_by_stress.values,
    color=[colors.get(x, '#808080') for x in power_by_stress.index]
)
ax4.set_xticks(range(len(power_by_stress)))
ax4.set_xticklabels(power_by_stress.index)
ax4.set_ylabel('Total Power Capacity (GW)')
ax4.set_title('Power Capacity by Water Stress Level', fontsize=12, fontweight='bold')
# Add value labels
for bar, val in zip(bars, power_by_stress.values):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f} GW', ha='center', fontsize=9)

# 5. Geographic Distribution by State
ax5 = plt.subplot(3, 3, 5)
state_counts = us_ai_datacenters['STATE_NAME'].value_counts().head(10)
bars = ax5.barh(range(len(state_counts)), state_counts.values, color='teal')
ax5.set_yticks(range(len(state_counts)))
ax5.set_yticklabels(state_counts.index, fontsize=9)
ax5.set_xlabel('Number of AI Data Centers')
ax5.set_title('Top 10 States with AI Data Centers', fontsize=12, fontweight='bold')
ax5.invert_yaxis()
# Add value labels
for i, (bar, val) in enumerate(zip(bars, state_counts.values)):
    ax5.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)

# 6. High-Risk Providers
ax6 = plt.subplot(3, 3, 6)
high_risk_data = us_ai_datacenters[us_ai_datacenters['water_stress_level'] != 'Low']
if len(high_risk_data) > 0:
    risk_providers = high_risk_data['PROVIDER_NAME'].value_counts().head(10)
    bars = ax6.barh(range(len(risk_providers)), risk_providers.values, color='coral')
    ax6.set_yticks(range(len(risk_providers)))
    ax6.set_yticklabels(risk_providers.index, fontsize=9)
    ax6.set_xlabel('Number of Facilities at Risk')
    ax6.set_title('Top Providers with AI Data Centers in High Water Stress', fontsize=12, fontweight='bold')
    ax6.invert_yaxis()
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, risk_providers.values)):
        ax6.text(val + 0.2, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
else:
    ax6.text(0.5, 0.5, 'No high-risk data centers found', ha='center', va='center')
    ax6.set_title('Top Providers in High Water Stress Areas', fontsize=12, fontweight='bold')

# 7. Timeline Analysis (if activation dates available)
ax7 = plt.subplot(3, 3, 7)
# Convert activation dates to datetime
us_ai_datacenters['ESTIMATED_ACTIVE_DATE'] = pd.to_datetime(
    us_ai_datacenters['DATA_CENTER_ACTIVATION_DATE'], 
    errors='coerce'
)
# If no activation date, use estimated date
us_ai_datacenters.loc[us_ai_datacenters['ESTIMATED_ACTIVE_DATE'].isna(), 'ESTIMATED_ACTIVE_DATE'] = pd.to_datetime(
    us_ai_datacenters.loc[us_ai_datacenters['ESTIMATED_ACTIVE_DATE'].isna(), 'DATA_CENTER_CONSTRUCTION_FINISHED_DATE'],
    errors='coerce'
)

# Group by year and water stress
future_data = us_ai_datacenters[us_ai_datacenters['ESTIMATED_ACTIVE_DATE'] >= pd.Timestamp('2025-01-01')]
if len(future_data) > 0:
    future_data['year'] = future_data['ESTIMATED_ACTIVE_DATE'].dt.year
    timeline_pivot = future_data.groupby(['year', 'water_stress_level']).size().unstack(fill_value=0)
    timeline_pivot.plot(kind='bar', stacked=True, ax=ax7, 
                        color=[colors.get(x, '#808080') for x in timeline_pivot.columns])
    ax7.set_xlabel('Year')
    ax7.set_ylabel('Number of Facilities')
    ax7.set_title('Planned AI Data Centers by Year and Water Stress', fontsize=12, fontweight='bold')
    ax7.legend(title='Water Stress', loc='upper left', fontsize=9)
    ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')
else:
    ax7.text(0.5, 0.5, 'No future activation dates available', ha='center', va='center')
    ax7.set_title('Planned AI Data Centers Timeline', fontsize=12, fontweight='bold')

# 8. Water Stress Zones Breakdown
ax8 = plt.subplot(3, 3, 8)
zone_counts = us_ai_datacenters[us_ai_datacenters['water_stress_zone'] != 'Low/None']['water_stress_zone'].value_counts()
if len(zone_counts) > 0:
    # Shorten zone names for display
    zone_labels = [z.split(' - ')[0] if ' - ' in z else z for z in zone_counts.index]
    bars = ax8.bar(range(len(zone_counts)), zone_counts.values, color='indianred')
    ax8.set_xticks(range(len(zone_counts)))
    ax8.set_xticklabels(zone_labels, rotation=45, ha='right', fontsize=9)
    ax8.set_ylabel('Number of AI Data Centers')
    ax8.set_title('AI Data Centers by Water Stress Zone', fontsize=12, fontweight='bold')
    # Add value labels
    for bar, val in zip(bars, zone_counts.values):
        ax8.text(bar.get_x() + bar.get_width()/2, val + 0.5, str(val), ha='center', fontsize=9)
else:
    ax8.text(0.5, 0.5, 'No data centers in water stress zones', ha='center', va='center')
    ax8.set_title('AI Data Centers by Water Stress Zone', fontsize=12, fontweight='bold')

# 9. Company Type Analysis
ax9 = plt.subplot(3, 3, 9)
# Analyze public vs private providers
us_ai_datacenters['is_disclosed'] = us_ai_datacenters['PROVIDER_NAME'] != 'Company Not Disclosed'
disclosure_counts = us_ai_datacenters.groupby(['is_disclosed', 'water_stress_level']).size().unstack(fill_value=0)
disclosure_counts.index = ['Undisclosed Provider', 'Named Provider']
disclosure_counts.plot(kind='bar', ax=ax9, color=[colors.get(x, '#808080') for x in disclosure_counts.columns])
ax9.set_xlabel('')
ax9.set_ylabel('Number of Facilities')
ax9.set_title('Provider Disclosure vs Water Stress', fontsize=12, fontweight='bold')
ax9.legend(title='Water Stress', loc='upper right', fontsize=9)
ax9.set_xticklabels(ax9.get_xticklabels(), rotation=0)

# Main title
fig.suptitle('AI Data Centers and Water Stress - Comprehensive Analysis', 
            fontsize=18, fontweight='bold', y=0.98)

# Add subtitle with key statistics
subtitle = f'Analysis of {len(us_ai_datacenters):,} US AI Data Centers • {total_at_risk:,} ({total_at_risk/len(us_ai_datacenters)*100:.1f}%) in High Water Stress Areas'
fig.text(0.5, 0.95, subtitle, ha='center', fontsize=12, color='#555555', style='italic')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.94])

# Save the figure
output_file = 'ai_datacenter_water_stress_analysis_charts.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Analysis charts saved as: {output_file}")

# Create detailed summary statistics
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)

print("\n1. Water Stress Exposure by Provider (Top 20):")
provider_stress = us_ai_datacenters.groupby('PROVIDER_NAME')['water_stress_level'].apply(
    lambda x: (x != 'Low').sum()
).sort_values(ascending=False).head(20)
for provider, count in provider_stress.items():
    total = len(us_ai_datacenters[us_ai_datacenters['PROVIDER_NAME'] == provider])
    pct = count/total*100 if total > 0 else 0
    print(f"  • {provider}: {count}/{total} ({pct:.1f}%) at risk")

print("\n2. Power Capacity at Risk by Zone:")
for zone in high_water_stress_zones.keys():
    zone_data = us_ai_datacenters[us_ai_datacenters['water_stress_zone'] == zone]
    if len(zone_data) > 0:
        power = zone_data['SELECTED_POWER_CAPACITY_MW'].sum()
        count = len(zone_data)
        print(f"  • {zone}: {count} facilities, {power:,.0f} MW")

print("\n3. State-Level Risk Assessment:")
state_risk = us_ai_datacenters.groupby('STATE_NAME').apply(
    lambda x: pd.Series({
        'total': len(x),
        'at_risk': (x['water_stress_level'] != 'Low').sum(),
        'pct_risk': (x['water_stress_level'] != 'Low').sum() / len(x) * 100 if len(x) > 0 else 0
    })
).sort_values('at_risk', ascending=False).head(10)

for state, data in state_risk.iterrows():
    if data['at_risk'] > 0:
        print(f"  • {state}: {int(data['at_risk'])}/{int(data['total'])} ({data['pct_risk']:.1f}%) at risk")

# Save detailed analysis to Excel
with pd.ExcelWriter('ai_datacenter_water_stress_detailed_analysis.xlsx', engine='openpyxl') as writer:
    # Overall summary
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total US AI Data Centers',
            'Data Centers in High Water Stress',
            'Data Centers in Extremely High Stress',
            'Percentage at Risk',
            'Total Power Capacity (MW)',
            'Power Capacity at Risk (MW)',
            'Number of Unique Providers',
            'Providers with Facilities at Risk'
        ],
        'Value': [
            len(us_ai_datacenters),
            total_at_risk,
            len(us_ai_datacenters[us_ai_datacenters['water_stress_level'] == 'Extremely High']),
            f"{total_at_risk/len(us_ai_datacenters)*100:.1f}%",
            f"{us_ai_datacenters['SELECTED_POWER_CAPACITY_MW'].sum():,.0f}",
            f"{us_ai_datacenters[us_ai_datacenters['water_stress_level'] != 'Low']['SELECTED_POWER_CAPACITY_MW'].sum():,.0f}",
            us_ai_datacenters['PROVIDER_NAME'].nunique(),
            us_ai_datacenters[us_ai_datacenters['water_stress_level'] != 'Low']['PROVIDER_NAME'].nunique()
        ]
    })
    summary_stats.to_excel(writer, sheet_name='Summary', index=False)
    
    # Provider analysis
    provider_analysis = us_ai_datacenters.groupby('PROVIDER_NAME').apply(
        lambda x: pd.Series({
            'Total Facilities': len(x),
            'Facilities at Risk': (x['water_stress_level'] != 'Low').sum(),
            'Percent at Risk': (x['water_stress_level'] != 'Low').sum() / len(x) * 100 if len(x) > 0 else 0,
            'Total Power (MW)': x['SELECTED_POWER_CAPACITY_MW'].sum(),
            'Power at Risk (MW)': x[x['water_stress_level'] != 'Low']['SELECTED_POWER_CAPACITY_MW'].sum()
        })
    ).sort_values('Facilities at Risk', ascending=False)
    provider_analysis.to_excel(writer, sheet_name='Provider Analysis')
    
    # State analysis
    state_analysis = us_ai_datacenters.groupby('STATE_NAME').apply(
        lambda x: pd.Series({
            'Total Facilities': len(x),
            'Facilities at Risk': (x['water_stress_level'] != 'Low').sum(),
            'Percent at Risk': (x['water_stress_level'] != 'Low').sum() / len(x) * 100 if len(x) > 0 else 0,
            'Total Power (MW)': x['SELECTED_POWER_CAPACITY_MW'].sum(),
            'Power at Risk (MW)': x[x['water_stress_level'] != 'Low']['SELECTED_POWER_CAPACITY_MW'].sum()
        })
    ).sort_values('Facilities at Risk', ascending=False)
    state_analysis.to_excel(writer, sheet_name='State Analysis')
    
    # Zone details
    zone_details = []
    for zone_name, zone_info in high_water_stress_zones.items():
        zone_data = us_ai_datacenters[us_ai_datacenters['water_stress_zone'] == zone_name]
        if len(zone_data) > 0:
            zone_details.append({
                'Zone': zone_name,
                'Stress Level': zone_info['label'],
                'Facilities': len(zone_data),
                'Power (MW)': zone_data['SELECTED_POWER_CAPACITY_MW'].sum(),
                'Top Provider': zone_data['PROVIDER_NAME'].value_counts().index[0] if len(zone_data) > 0 else 'N/A',
                'Top State': zone_data['STATE_NAME'].value_counts().index[0] if len(zone_data) > 0 else 'N/A'
            })
    if zone_details:
        pd.DataFrame(zone_details).to_excel(writer, sheet_name='Zone Analysis', index=False)

print(f"\nDetailed analysis saved to: ai_datacenter_water_stress_detailed_analysis.xlsx")

# Show the charts
plt.show()
print("\nAnalysis complete!")
