import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def parse_score(score_str):
    """Extract score and confidence intervals from string like '0.0223 (0.0206, 0.0241)'"""
    pattern = r'(-?\d+\.\d+)\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)'
    match = re.match(pattern, score_str)
    if match:
        score = float(match.group(1))
        ci_low = float(match.group(2))
        ci_high = float(match.group(3))
        return score, ci_low, ci_high
    return None, None, None

def get_model_color(name):
    """Return color based on model name"""
    colors = {
        'claude': '#4A90E2',
        'gpt': '#45B764',
        'gemini': '#E6A040',
        'mistral': '#9B6B9E',
        'deepseek': '#E57373'
    }
    return next((v for k, v in colors.items() if k in name.lower()), '#757575')

def create_model_chart(data, filename, figsize=(4, 4)):
    """Create bar chart for given model data"""
    # Adjust figure size for all models chart if needed
    if len(data) > 5:
        figsize = (8, 4)
        
    plt.figure(figsize=figsize)
    
    # Extract data
    scores = []
    yerr_low = []
    yerr_high = []
    names = []
    colors = []
    
    for _, row in data.iterrows():
        score, ci_low, ci_high = parse_score(row['Score'])
        if score is not None:
            scores.append(score)
            yerr_low.append(score - ci_low)
            yerr_high.append(ci_high - score)
            names.append(row['Model'])
            colors.append(get_model_color(row['Model']))
    
    # Create plot background with grid
    plt.grid(True, axis='y', alpha=0.3, zorder=0)
    
    # Add black gridline at zero
    plt.axhline(y=0, color='black', linewidth=1.0, zorder=2)
    
    # Create bar plot
    x = np.arange(len(names))
    bars = plt.bar(x, scores, color=colors, width=0.8, zorder=3)
    
    # Add error bars on top
    plt.errorbar(x, scores, yerr=[yerr_low, yerr_high], fmt='none', color='gray', capsize=5, zorder=4)
    
    # Format axes
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Score')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Hardcoded data from the CSV
    data = {
        'Model': [
            'Mistral-Large',
            'Gemini-1.5-Pro',
            'Gemini-1.5-Flash',
            'Claude-3-Opus',
            'DeepSeek-v3',
            'Claude-3.5-Haiku',
            'Claude-3.5-Sonnet',            
            'GPT-4o',            
            'GPT-4o-Mini',
            'Llama-3.3'
        ],
        'Score': [
            '0.0682 (0.0474, 0.0890)',
            '0.0659  (0.0591, 0.0727)',
            '0.0501  (0.0354, 0.0648)',
            '0.0429  (0.0287, 0.0571)',
            '0.0395  (0.0186, 0.0604)',
            '0.0198  (0.0052, 0.0344)',
            '0.0180  (0.0113, 0.0246)',
            '0.0110 (0.0040, 0.0180)',
            '0.0017 (-0.0139, 0.0173)',
            '-0.0149 (-0.0363, 0.0065)',
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create all models chart first
    create_model_chart(df, 'all_models.png', figsize=(10, 5))
    
    # Create three separate charts
    # Large models
    large_models = df[df['Model'].isin(['Claude-3-Opus','Claude-3.5-Sonnet', 'Gemini-1.5-Pro', 'GPT-4o'])]
    create_model_chart(large_models, 'large_models.png')
    
    # Compact models
    compact_models = df[df['Model'].isin(['Claude-3.5-Haiku', 'Gemini-1.5-Flash', 'GPT-4o-Mini'])]
    create_model_chart(compact_models, 'compact_models.png')
    
    # Open models
    open_models = df[df['Model'].isin(['DeepSeek-v3', 'Llama-3.3', 'Mistral-Large'])]
    create_model_chart(open_models, 'open_models.png')

if __name__ == '__main__':
    main()