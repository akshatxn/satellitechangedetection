import matplotlib.pyplot as plt
import numpy as np

def generate_presentation_chart():
    # ---------------------------------------------------------
    # 1. YOUR RESEARCH DATA (Simulated from actual Ground Truth Labels)
    # We maintain the sorting as found visually, not numerical sort.
    # Prague (1.90%) is listed before Rome (1.97%) in this simulation.
    # ---------------------------------------------------------
    cities = ['Abu Dhabi', 'Dubai', 'Berlin_Outskirts', 'Madrid', 'Paris_Suburb', 
              'Prague', 'Rome', 'Riyadh', 'Sao_Paulo', 'Valencia', 'Milano']
    
    # Growth percentage values as provided in the simulation
    growth_values = [5.21, 4.15, 3.88, 3.11, 2.72, 
                     1.90, 1.97, 1.64, 0.49, 0.33, 0.45]

    # ---------------------------------------------------------
    # 2. PPT FORMATTING SETUP
    # We use a white background to perfectly merge with PowerPoint slides.
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7), facecolor='white')
    ax = plt.gca()
    
    # --- Modern Visual Style ---
    # We remove the top and right black borders (spines) for a clean look.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # We set discrete horizontal gridlines for easy comparison.
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#cccccc')
    
    # We create a simple blue colormap for visual impact (Dark = High Growth).
    cmap = plt.get_cmap('Blues')
    
    # ---------------------------------------------------------
    # 3. CREATING THE BARS
    # We use rank-based coloring (not normalized to values) to replicate the look.
    # Colors transition linearly from Dark Blue to Light Blue.
    # ---------------------------------------------------------
    # Create the gradient of colors
    colors = cmap(np.linspace(1.0, 0.4, len(cities)))
    
    # Generate the bar chart
    bars = plt.bar(cities, growth_values, color=colors, edgecolor='black', linewidth=1.1)

    # ---------------------------------------------------------
    # 4. AXIS CUSTOMIZATION
    # ---------------------------------------------------------
    # --- X-Axis (Cities) ---
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    # --- Y-Axis (Growth %) ---
    ax.set_ylim(0, 6)  # We cap the y-axis at 6% for presentation clarity.
    
    # We format the labels to add the '%' symbol clearly.
    # We create custom ticks at 0, 1, 2, 3, 4, 5, 6
    vals = np.arange(0, 7)
    ax.set_yticks(vals)
    
    # Generate the labels like '0%', '1%' ... '6%'
    ax.set_yticklabels([f'{val:.0f}%' for val in vals], fontsize=11)
    
    # Set the bold, modern axis label.
    ax.set_ylabel("% Area Changed (Ground Truth)", fontsize=14, fontweight='bold', labelpad=15)

    # ---------------------------------------------------------
    # 5. DATA LABELS (On top of bars)
    # We add the exact percentages on top of each bar so the values are instantly readable.
    # ---------------------------------------------------------
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.12, 
                 f'{height:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # ---------------------------------------------------------
    # 6. CHART TITLE & SAVING
    # ---------------------------------------------------------
    plt.title('URBAN GROWTH COMPARISON BY CITY (ACTUAL AREA CHANGED)', 
              fontsize=18, fontweight='bold', pad=30, color='black')
    
    # Adjust layout so labels are not cut off.
    plt.tight_layout()
    
    # Save the chart as a high-resolution PNG, perfect for dragging into PPT.
    filename = 'urban_growth_chart.png'
    print(f"\n📊 Generated modern comparison chart successfully...")
    print(f"✅ Image saved as: '{filename}'")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    
    # plt.show() # Uncomment to see the chart immediately while testing.

if __name__ == "__main__":
    generate_presentation_chart()