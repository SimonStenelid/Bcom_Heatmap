#!/usr/bin/env python3
"""
Bot Detection Heatmap Generator for Booking.com Flight Clicks

Analyzes click data to identify bot traffic and generates a heatmap showing
bot score per site/market.

Bot Detection Criteria:
- User agent must have >= 300 entry clicks
- Conversion rate < 1%

Bot Score Calculation:
- Bot Score = (Total clicks from bot user agents / Total clicks for site) * 100
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class BotDetector:
    def __init__(self, min_clicks=300, max_conversion_rate=1.0):
        """
        Initialize bot detector with thresholds.

        Args:
            min_clicks: Minimum clicks required to classify as potential bot
            max_conversion_rate: Maximum conversion rate % to classify as bot
        """
        self.min_clicks = min_clicks
        self.max_conversion_rate = max_conversion_rate

    def analyze_data(self, df):
        """
        Analyze data to identify bots and calculate bot scores with volume weighting.

        Args:
            df: DataFrame with columns [Site, User_Agent, Entry_Clicks, Orders]

        Returns:
            tuple: (bot_df, site_scores, summary_stats)
        """
        # Calculate conversion rate for each row
        df['Conversion_Rate'] = (df['Orders'] / df['Entry_Clicks'] * 100).fillna(0)

        # Identify bot user agents
        df['Is_Bot'] = (
            (df['Entry_Clicks'] >= self.min_clicks) &
            (df['Conversion_Rate'] < self.max_conversion_rate)
        )

        # Get bot data
        bot_df = df[df['Is_Bot']].copy()

        # Calculate volume weight for each bot user agent
        # Higher click volumes get exponentially higher weights
        # Weight = log10(clicks / min_clicks + 1) to emphasize high-volume bots
        bot_df['Volume_Weight'] = np.log10(bot_df['Entry_Clicks'] / self.min_clicks + 1)
        bot_df['Weighted_Clicks'] = bot_df['Entry_Clicks'] * bot_df['Volume_Weight']

        # Calculate weighted bot score per site
        site_total_clicks = df.groupby('Site')['Entry_Clicks'].sum()
        site_bot_clicks = bot_df.groupby('Site')['Entry_Clicks'].sum()
        site_weighted_bot_clicks = bot_df.groupby('Site')['Weighted_Clicks'].sum()

        site_scores = pd.DataFrame({
            'Total_Clicks': site_total_clicks,
            'Bot_Clicks': site_bot_clicks,
            'Weighted_Bot_Clicks': site_weighted_bot_clicks
        }).fillna(0)

        # Normalize weighted clicks to a 0-100 scale per site
        # This gives us a weighted bot score where high-volume bots have more impact
        max_possible_weight = np.log10(site_scores['Total_Clicks'] / self.min_clicks + 1)
        weighted_bot_score = (
            (site_scores['Weighted_Bot_Clicks'] / (site_scores['Total_Clicks'] * max_possible_weight)) * 100
        ).fillna(0)

        # Calculate raw percentage
        raw_bot_pct = (site_scores['Bot_Clicks'] / site_scores['Total_Clicks'] * 100).fillna(0)

        # Use the maximum of weighted and raw score to ensure high-volume bots are emphasized
        site_scores['Bot_Score'] = pd.DataFrame({
            'weighted': weighted_bot_score,
            'raw': raw_bot_pct
        }).max(axis=1).round(2)

        # Count bot user agents per site
        bot_count_per_site = bot_df.groupby('Site').size()
        site_scores['Bot_UA_Count'] = bot_count_per_site
        site_scores['Bot_UA_Count'] = site_scores['Bot_UA_Count'].fillna(0).astype(int)

        # Summary statistics
        summary_stats = {
            'total_rows': len(df),
            'total_clicks': df['Entry_Clicks'].sum(),
            'total_orders': df['Orders'].sum(),
            'overall_conversion': df['Orders'].sum() / df['Entry_Clicks'].sum() * 100,
            'bot_user_agents': len(bot_df),
            'bot_clicks': bot_df['Entry_Clicks'].sum(),
            'bot_clicks_pct': bot_df['Entry_Clicks'].sum() / df['Entry_Clicks'].sum() * 100,
            'sites_with_bots': len(site_scores[site_scores['Bot_Score'] > 0])
        }

        return bot_df, site_scores, summary_stats

    def generate_heatmap(self, site_scores, output_path='bot_heatmap.png'):
        """
        Generate heatmap visualization of bot scores by site/market in a grid layout.

        Args:
            site_scores: DataFrame with bot scores per site
            output_path: Path to save the PNG image
        """
        # Sort sites by bot score descending
        site_scores_sorted = site_scores.sort_values('Bot_Score', ascending=False)

        # Get site names and scores
        sites = site_scores_sorted.index.tolist()
        scores = site_scores_sorted['Bot_Score'].tolist()

        num_sites = len(sites)

        # Calculate grid dimensions (make it as square as possible)
        cols = int(np.ceil(np.sqrt(num_sites)))
        rows = int(np.ceil(num_sites / cols))

        # Pad with NaN to fill the grid
        total_cells = rows * cols
        sites_padded = sites + [''] * (total_cells - num_sites)
        scores_padded = scores + [np.nan] * (total_cells - num_sites)

        # Reshape into grid
        grid_data = np.array(scores_padded).reshape(rows, cols)
        grid_labels = np.array(sites_padded).reshape(rows, cols)

        # Create custom annotations with site names, scores, and total clicks
        annot_labels = np.empty((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                if grid_labels[i, j] and not np.isnan(grid_data[i, j]):
                    site_name = grid_labels[i, j].replace('booking_', '').upper()
                    site_key = grid_labels[i, j]
                    total_clicks = site_scores.loc[site_key, 'Total_Clicks']

                    # Format clicks with K/M suffix for readability
                    if total_clicks >= 1_000_000:
                        clicks_str = f'{total_clicks/1_000_000:.1f}M'
                    elif total_clicks >= 1_000:
                        clicks_str = f'{total_clicks/1_000:.1f}K'
                    else:
                        clicks_str = f'{int(total_clicks)}'

                    annot_labels[i, j] = f'{site_name}\n{grid_data[i, j]:.1f}%\n{clicks_str}'
                else:
                    annot_labels[i, j] = ''

        # Calculate figure size based on grid dimensions
        cell_size = 1.5
        fig_width = cols * cell_size + 2
        fig_height = rows * cell_size + 1.5

        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Generate heatmap
        sns.heatmap(
            grid_data,
            annot=annot_labels,
            fmt='',
            cmap='RdYlGn_r',  # Red (high bot) -> Yellow -> Green (low bot)
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Bot Score (%)', 'shrink': 0.8},
            linewidths=2,
            linecolor='white',
            square=True,
            ax=ax,
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )

        # Remove axis labels and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title with yesterday's date (since data represents previous complete day)
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        ax.set_title(
            f'Booking Bot Map ({yesterday})',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Heatmap saved to: {output_path}")

        plt.close()

    def print_summary(self, summary_stats, bot_df, site_scores):
        """Print summary statistics to console."""
        print("\n" + "="*70)
        print("BOT DETECTION SUMMARY")
        print("="*70)

        print(f"\nDetection Criteria:")
        print(f"  - Minimum clicks: {self.min_clicks:,}")
        print(f"  - Maximum conversion rate: {self.max_conversion_rate}%")

        print(f"\nOverall Statistics:")
        print(f"  - Total data rows: {summary_stats['total_rows']:,}")
        print(f"  - Total clicks: {summary_stats['total_clicks']:,}")
        print(f"  - Total orders: {summary_stats['total_orders']:,}")
        print(f"  - Overall conversion rate: {summary_stats['overall_conversion']:.2f}%")

        print(f"\nBot Detection Results:")
        print(f"  - Bot user agents identified: {summary_stats['bot_user_agents']:,}")
        print(f"  - Bot clicks: {summary_stats['bot_clicks']:,}")
        print(f"  - Bot clicks percentage: {summary_stats['bot_clicks_pct']:.2f}%")
        print(f"  - Sites with bot traffic: {summary_stats['sites_with_bots']}")

        print(f"\nTop 10 Sites by Bot Score:")
        print("-" * 70)
        top_sites = site_scores.sort_values('Bot_Score', ascending=False).head(10)
        for idx, (site, row) in enumerate(top_sites.iterrows(), 1):
            print(f"  {idx:2d}. {site:20s} | Bot Score: {row['Bot_Score']:5.1f}% | "
                  f"Bot UAs: {int(row['Bot_UA_Count']):3d} | "
                  f"Bot Clicks: {int(row['Bot_Clicks']):,}")

        print(f"\nTop 10 Bot User Agents by Click Volume:")
        print("-" * 70)
        top_bots = bot_df.sort_values('Entry_Clicks', ascending=False).head(10)
        for idx, (_, row) in enumerate(top_bots.iterrows(), 1):
            print(f"  {idx:2d}. {row['User_Agent'][:50]:50s} | "
                  f"Clicks: {row['Entry_Clicks']:,} | "
                  f"Conv: {row['Conversion_Rate']:.2f}%")

        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate bot detection heatmap from Booking.com flight click data'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input Excel file (.xlsx)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='bot_heatmap.png',
        help='Output path for heatmap image (default: bot_heatmap.png)'
    )
    parser.add_argument(
        '--min-clicks',
        type=int,
        default=300,
        help='Minimum clicks to classify as potential bot (default: 300)'
    )
    parser.add_argument(
        '--max-conversion',
        type=float,
        default=1.0,
        help='Maximum conversion rate %% to classify as bot (default: 1.0)'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() != '.xlsx':
        print(f"Error: Input file must be .xlsx format", file=sys.stderr)
        sys.exit(1)

    # Read data
    print(f"Reading data from: {args.input_file}")
    try:
        df = pd.read_excel(args.input_file)
    except Exception as e:
        print(f"Error reading Excel file: {e}", file=sys.stderr)
        sys.exit(1)

    # Clean column names
    df.columns = ['Site', 'User_Agent', 'Entry_Clicks', 'Orders']

    print(f"Loaded {len(df):,} rows")

    # Filter out rows with null/empty User_Agent
    initial_count = len(df)
    df = df[df['User_Agent'].notna() & (df['User_Agent'].str.strip() != '')]
    filtered_count = initial_count - len(df)

    if filtered_count > 0:
        print(f"Filtered out {filtered_count:,} rows with null/empty User Agent")

    if len(df) == 0:
        print("Error: No valid data remaining after filtering", file=sys.stderr)
        sys.exit(1)

    # Initialize detector
    detector = BotDetector(
        min_clicks=args.min_clicks,
        max_conversion_rate=args.max_conversion
    )

    # Analyze data
    print("Analyzing data for bot patterns...")
    bot_df, site_scores, summary_stats = detector.analyze_data(df)

    # Print summary
    detector.print_summary(summary_stats, bot_df, site_scores)

    # Generate heatmap
    print("Generating heatmap...")
    detector.generate_heatmap(site_scores, output_path=args.output)

    print("✓ Done!")


if __name__ == '__main__':
    main()
