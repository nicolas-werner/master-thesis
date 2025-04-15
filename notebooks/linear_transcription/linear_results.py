import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    import sys
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import json
    import base64
    import io

    # Add project root to path to allow imports from src
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Switch to non-interactive backend for matplotlib
    plt.switch_backend('Agg')
    return base64, io, json, mo, np, os, pd, plt, project_root, sns, sys


@app.cell
def _(mo):
    mo.md(
        r"""
        # Linear Transcription Results Comparison

        This notebook compares the performance of different transcription methods.
        """
    )
    return


@app.cell
def display_comparison_stats(base64, io, np, plt, sns):
    from src.visualization import plot_method_comparison, plot_error_type_breakdown, plot_metric_distribution
    from src.file_utils import extract_text_from_xml

    def display_comparison_stats(results, transkribus_df, mo, pd, provider_suffix=""):
        """
        Display comparison stats between models and Transkribus with visualizations

        Args:
            results: Results from run_evaluation
            transkribus_df: DataFrame with Transkribus results
            mo: Marimo module
            pd: Pandas module
            provider_suffix: Optional suffix to add to provider names (for hybrid models)

        Returns:
            Marimo UI component with stats, visualizations, and the comparison dataframe
        """
        # Store Transkribus metrics for comparison
        transkribus_metrics = {
            "avg_cer": transkribus_df['cer'].mean() if not transkribus_df.empty else None,
            "avg_wer": transkribus_df['wer'].mean() if not transkribus_df.empty else None,
            "avg_bwer": transkribus_df['bwer'].mean() if not transkribus_df.empty else None
        }

        # Create stat components for comparison
        final_provider_stats = []

        # Add Transkribus baseline stats at the top
        if not transkribus_df.empty:
            transkribus_stats = [
                mo.md("### TRANSKRIBUS with Text Titan 1 (Baseline)"),
                mo.hstack([
                    mo.stat(f"{transkribus_metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{transkribus_metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{transkribus_metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                    mo.stat(len(transkribus_df), label="Documents", bordered=True)
                ])
            ]
        else:
            transkribus_stats = [mo.md("### Transkribus baseline not available")]

        # Create comparison stats for each provider
        for provider_name, metrics in results["comparison_data"].items():
            display_name = f"{provider_name.upper()}{provider_suffix}"
            _model_name = metrics["model"]

            # Compare with Transkribus metrics and determine direction
            if transkribus_metrics["avg_cer"] is not None:
                # Calculate absolute differences
                cer_diff = transkribus_metrics["avg_cer"] - metrics["avg_cer"]
                wer_diff = transkribus_metrics["avg_wer"] - metrics["avg_wer"]
                bwer_diff = transkribus_metrics["avg_bwer"] - metrics["avg_bwer"]

                # Calculate percentage changes (relative to Transkribus baseline)
                cer_pct = (cer_diff / transkribus_metrics["avg_cer"]) * 100 if transkribus_metrics["avg_cer"] > 0 else 0
                wer_pct = (wer_diff / transkribus_metrics["avg_wer"]) * 100 if transkribus_metrics["avg_wer"] > 0 else 0
                bwer_pct = (bwer_diff / transkribus_metrics["avg_bwer"]) * 100 if transkribus_metrics["avg_bwer"] > 0 else 0

                # For error metrics like CER/WER/BWER, lower is better
                # So if our model has lower error (cer_diff > 0), that's an improvement
                cer_direction = "increase" if cer_diff > 0 else "decrease"
                wer_direction = "increase" if wer_diff > 0 else "decrease"
                bwer_direction = "increase" if bwer_diff > 0 else "decrease"

                # Format percentage with sign (positive means improvement)
                cer_caption = f"{cer_pct:+.1f}% vs Transkribus"
                wer_caption = f"{wer_pct:+.1f}% vs Transkribus"
                bwer_caption = f"{bwer_pct:+.1f}% vs Transkribus"

                # Create stat components row for this provider
                final_provider_stats.append(mo.md(f"#### {display_name} with {_model_name}"))
                final_provider_stats.append(
                    mo.hstack([
                        mo.stat(
                            f"{metrics['avg_cer']:.4f}",
                            label="Average CER",
                            caption=cer_caption,
                            direction=cer_direction,
                            bordered=True
                        ),
                        mo.stat(
                            f"{metrics['avg_wer']:.4f}",
                            label="Average WER",
                            caption=wer_caption,
                            direction=wer_direction,
                            bordered=True
                        ),
                        mo.stat(
                            f"{metrics['avg_bwer']:.4f}",
                            label="Average BWER",
                            caption=bwer_caption,
                            direction=bwer_direction,
                            bordered=True
                        ),
                        mo.stat(metrics["doc_count"], label="Documents", bordered=True)
                    ])
                )
            else:
                # If Transkribus metrics aren't available, show regular stats
                final_provider_stats.append(mo.md(f"#### {display_name} with {_model_name}"))
                final_provider_stats.append(
                    mo.hstack([
                        mo.stat(f"{metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                        mo.stat(f"{metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                        mo.stat(f"{metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                        mo.stat(metrics["doc_count"], label="Documents", bordered=True)
                    ])
                )

        # Return both the complete UI and comparison dataframe
        return mo.vstack([
            mo.md("### Model Performance vs Transkribus Baseline"),
            *transkribus_stats,
            *final_provider_stats
        ]), results["comparison_data"]

    def create_cross_method_comparison(all_results_dict, transkribus_df, mo, pd):
        """
        Create comprehensive visualizations comparing performance across all methods and providers

        Args:
            all_results_dict: Dictionary mapping evaluation types to their respective results
                              e.g. {'zero_shot': zero_shot_results, 'one_shot': one_shot_results, ...}
            transkribus_df: DataFrame with Transkribus results
            mo: Marimo module
            pd: Pandas module

        Returns:
            Marimo UI component with visualizations
        """
        # Define method order and scientific color palette for consistent visualization
        method_order = [
            'Baseline',
            'Zero Shot',
            'Hybrid Zero Shot',
            'One Shot',
            'One Shot Hybrid',
            'Two Shot',
            'Two Shot Hybrid'
        ]

        # Scientific color palette (colorblind-friendly)
        provider_colors = {
            'openai': '#1f77b4',      # Blue
            'gemini': '#2ca02c',      # Green
            'mistral': '#d62728',     # Red
            'Transkribus': '#7f7f7f'  # Gray
        }

        # Prepare unified dataframe for all results
        all_method_results = []

        # Add Transkribus baseline
        if not transkribus_df.empty:
            for _, row in transkribus_df.iterrows():
                result_row = {
                    'document_id': row['document_id'],
                    'provider': 'Transkribus',
                    'method': 'Baseline',
                    'model': 'Text Titan 1',
                    'cer': row['cer'],
                    'wer': row['wer'],
                    'bwer': row['bwer']
                }
                all_method_results.append(result_row)

        # Process each evaluation type and its results
        for eval_type, results in all_results_dict.items():
            if not results or not isinstance(results, dict) or 'provider_results' not in results:
                continue

            # Format method name for display
            if eval_type == "zero_shot":
                display_method = "Zero Shot"
            elif eval_type == "hybrid_zero_shot":
                display_method = "Hybrid Zero Shot"
            elif eval_type == "one_shot":
                display_method = "One Shot"
            elif eval_type == "one_shot_hybrid":
                display_method = "One Shot Hybrid"
            elif eval_type == "two_shot":
                display_method = "Two Shot"
            elif eval_type == "two_shot_hybrid":
                display_method = "Two Shot Hybrid"
            else:
                display_method = eval_type.replace('_', ' ').title()

            # Add results for each provider in this evaluation type
            for provider, provider_df in results['provider_results'].items():
                model_name = results['comparison_data'][provider]['model']

                for _, doc_metrics in provider_df.iterrows():
                    result_row = {
                        'document_id': doc_metrics['document_id'],
                        'provider': provider,
                        'method': display_method,
                        'model': model_name,
                        'cer': doc_metrics['cer'],
                        'wer': doc_metrics['wer'],
                        'bwer': doc_metrics['bwer']
                    }
                    all_method_results.append(result_row)

        # Create the unified dataframe
        if not all_method_results:
            return mo.md("No results available for comparison")

        unified_df = pd.DataFrame(all_method_results)

        # Create a composite identifier for provider-method combinations
        unified_df['provider_method'] = unified_df['provider'] + ' (' + unified_df['method'] + ')'

        # Convert method column to categorical with specified order
        if 'method' in unified_df.columns:
            available_methods = [m for m in method_order if m in unified_df['method'].unique()]
            unified_df['method'] = pd.Categorical(
                unified_df['method'],
                categories=available_methods,
                ordered=True
            )

        # Method-wise comparison chart (grouped by method type)
        method_wise_chart = []

        # Define method groups and their display order (use the ones we actually have)
        method_groups = [m for m in method_order if m in unified_df['method'].unique()]

        # Get unique providers in the data
        providers = sorted(unified_df['provider'].unique())

        # Create a figure for method-wise comparison
        plt.figure(figsize=(18, 10))

        # Adjust the width based on number of providers
        width = 18 * len(method_groups) / 6

        plt.figure(figsize=(width, 10))

        # For each metric, create a row of method groups
        metrics = ['wer', 'cer', 'bwer']
        metric_titles = {
            'cer': 'Character Error Rate (CER)',
            'wer': 'Word Error Rate (WER)',
            'bwer': 'Bag-of-Words Error Rate (BWER)'
        }

        # Store baseline values for each metric to use in later charts
        baseline_values = {}

        for mi, metric in enumerate(metrics):
            plt.subplot(len(metrics), 1, mi+1)

            # Calculate positions for each group
            x_positions = []
            x_labels = []
            x_ticks = []

            current_pos = 0
            baseline_value = None

            for gi, method_group in enumerate(method_groups):
                method_providers = unified_df[unified_df['method'] == method_group]['provider'].unique()

                if len(method_providers) == 0:
                    continue

                # Add group label position
                center_pos = current_pos + (len(method_providers) / 2)
                x_ticks.append(center_pos)
                x_labels.append(method_group)

                # Plot each provider bar within this method group
                for pi, provider in enumerate(providers):
                    if provider in method_providers:
                        # Get data for this provider and method
                        provider_data = unified_df[(unified_df['provider'] == provider) &
                                                (unified_df['method'] == method_group)]

                        if len(provider_data) > 0:
                            # Calculate mean and std for error bars
                            mean_val = provider_data[metric].mean()
                            std_val = provider_data[metric].std() / np.sqrt(len(provider_data))  # Standard error

                            # Save baseline value for reference line
                            if method_group == 'Baseline' and provider == 'Transkribus':
                                baseline_value = mean_val
                                baseline_values[metric] = mean_val

                            # Plot bar
                            bar = plt.bar(current_pos, mean_val,
                                        width=0.8,
                                        color=provider_colors.get(provider, '#333333'),
                                        edgecolor='black',
                                        linewidth=0.5,
                                        yerr=std_val,
                                        capsize=5)

                            # Add value label without the line artifact
                            plt.text(current_pos, mean_val + std_val + 0.01,
                                   f"{mean_val:.3f}",
                                   ha='center',
                                   va='bottom',
                                   fontsize=9,
                                   bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))

                            # Store position for legend
                            x_positions.append(current_pos)

                            # Move to next position
                            current_pos += 1

                # Add space between groups
                current_pos += 1

            # Add reference line for baseline value if it exists
            if baseline_value is not None:
                plt.axhline(y=baseline_value, color='orange', linestyle=':', linewidth=1.5,
                           alpha=0.8, label=f'Baseline ({baseline_value:.3f})')

            # Set axis labels and title
            plt.ylabel(metric_titles[metric], fontsize=12)
            if mi == len(metrics) - 1:  # Only for the last subplot
                plt.xlabel('Method', fontsize=12)

            # Set tick positions and labels
            plt.xticks(x_ticks, x_labels, rotation=0, fontsize=10)

            # Add grid for readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Only add legend to the first subplot
            if mi == 0:
                # Create custom legend
                legend_elements = [plt.Rectangle((0,0), 1, 1,
                                              facecolor=provider_colors.get(provider, '#333333'),
                                              edgecolor='black',
                                              linewidth=0.5,
                                              label=provider.title())
                                 for provider in providers]

                # Add reference line to legend if baseline exists
                if baseline_value is not None:
                    legend_elements.append(plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=1.5,
                                                   label=f'Baseline ({baseline_value:.3f})'))

                plt.legend(handles=legend_elements,
                         title='Provider',
                         loc='upper right',
                         ncol=len(providers))

        plt.suptitle('Method-wise Comparison by Provider', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # method_wise_chart.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Method-wise Comparison" width="100%" />'))

        # Create a grid of metric visualizations
        metric_grid = []

        # Create a figure for all metric combinations
        plt.figure(figsize=(15, 15))

        # Create a 3x1 grid of bar plots for CER, WER, BWER by method and provider
        for i, metric in enumerate(['cer', 'wer', 'bwer']):
            plt.subplot(3, 1, i+1)

            # Calculate aggregate metrics with confidence intervals
            agg_df = unified_df.groupby(['method', 'provider'], observed=True)[metric].agg(['mean', 'count', 'std']).reset_index()
            agg_df['se'] = agg_df['std'] / np.sqrt(agg_df['count'])  # Standard error
            agg_df['ci'] = agg_df['se'] * 1.96  # 95% confidence interval

            # Create grouped bar plot
            ax = sns.barplot(x='method', y='mean', hue='provider', data=agg_df,
                           palette=provider_colors, errorbar=('ci', 95))

            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)

            # Add reference line for baseline value if it exists
            if metric in baseline_values:
                plt.axhline(y=baseline_values[metric], color='orange', linestyle=':', linewidth=1.5,
                           alpha=0.8)

            # Set labels and title
            plt.title(f'{metric.upper()} by Method and Provider')
            plt.ylabel(metric.upper())
            plt.xlabel('Method')
            plt.xticks(rotation=30, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            # Create legend with baseline included only once
            handles, labels = ax.get_legend_handles_labels()
            baseline_label = f'Baseline ({baseline_values[metric]:.3f})' if metric in baseline_values else None

            if baseline_label:
                # Add baseline line to handles and labels (only once)
                baseline_line = plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=1.5)
                handles.append(baseline_line)
                labels.append(baseline_label)

            # Create custom legend with unique entries
            ax.legend(handles=handles, labels=labels, title='Provider', loc='best')

            plt.tight_layout()

        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        metric_grid.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Performance Grid" width="100%" />'))

        # Create box plots to identify outliers
        boxplot_charts = []

        # Create box plots for each metric
        plt.figure(figsize=(18, 15))

        for i, metric in enumerate(['cer', 'wer', 'bwer']):
            plt.subplot(3, 1, i+1)

            # Create box plot
            ax = sns.boxplot(x='method', y=metric, hue='provider', data=unified_df, palette=provider_colors)

            # Add individual points to show outliers more clearly
            sns.stripplot(x='method', y=metric, hue='provider', data=unified_df, 
                        size=4, alpha=0.5, palette=provider_colors, dodge=True, 
                        jitter=True, linewidth=0)

            # Remove duplicate legend entries from stripplot
            handles, labels = ax.get_legend_handles_labels()
            unique_handles = handles[:len(providers)]
            ax.legend(unique_handles, providers, title="Provider", loc='best')

            # Set labels and title
            plt.title(f'{metric_titles[metric]} Distribution with Outliers')
            plt.ylabel(metric.upper())
            plt.xlabel('Method')
            plt.xticks(rotation=30, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            plt.tight_layout()

        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        boxplot_charts.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Boxplots with Outliers" width="100%" />'))

        # Create overall ranking chart with Transkribus in grey
        ranking_chart = []

        # Calculate means for each provider-method combination
        avg_metrics = unified_df.groupby('provider_method')[['cer', 'wer', 'bwer']].mean().reset_index()

        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(['cer', 'wer', 'bwer']):
            plt.subplot(3, 1, i+1)

            # Sort by metric value (ascending, since lower is better)
            sorted_df = avg_metrics.sort_values(by=metric)

            # Create a color map for the bars
            bar_colors = []
            for provider_method in sorted_df['provider_method']:
                if 'Transkribus' in provider_method:
                    bar_colors.append('#7f7f7f')  # Grey for Transkribus
                elif 'openai' in provider_method:
                    bar_colors.append('#1f77b4')  # Blue for OpenAI
                elif 'gemini' in provider_method:
                    bar_colors.append('#2ca02c')  # Green for Gemini
                elif 'mistral' in provider_method:
                    bar_colors.append('#d62728')  # Red for Mistral
                else:
                    bar_colors.append('#333333')  # Default dark grey

            # Create horizontal bar chart with proper colors
            bars = plt.barh(sorted_df['provider_method'], sorted_df[metric], color=bar_colors)

            # Add value labels
            for j, v in enumerate(sorted_df[metric]):
                plt.text(v + 0.005, j, f"{v:.3f}", va='center')

            plt.title(f'Ranking by {metric.upper()} (Lower is Better)')
            plt.xlabel(metric.upper())
            plt.ylabel('')

            # Add a legend for the provider colors
            legend_elements = [
                plt.Rectangle((0,0), 1, 1, facecolor='#7f7f7f', edgecolor='none', label='Transkribus'),
                plt.Rectangle((0,0), 1, 1, facecolor='#1f77b4', edgecolor='none', label='OpenAI'),
                plt.Rectangle((0,0), 1, 1, facecolor='#2ca02c', edgecolor='none', label='Gemini'),
                plt.Rectangle((0,0), 1, 1, facecolor='#d62728', edgecolor='none', label='Mistral')
            ]
            plt.legend(handles=legend_elements, loc='lower right')

            plt.tight_layout()

        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        ranking_chart.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Ranking Chart" width="100%" />'))

        # Create comprehensive comparison table with confidence intervals
        # Calculate aggregate metrics for each provider-method combination
        table_data = unified_df.groupby(['provider', 'method', 'model'], observed=True).agg({
            'cer': ['mean', 'std', 'min', 'max', 'count'],
            'wer': ['mean', 'std', 'min', 'max'],
            'bwer': ['mean', 'std', 'min', 'max'],
        }).reset_index()

        # Flatten multi-level columns
        table_data.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col
            for col in table_data.columns.values
        ]

        # Add confidence intervals
        table_data['cer_ci'] = table_data['cer_std'] / np.sqrt(table_data['cer_count']) * 1.96
        table_data['wer_ci'] = table_data['wer_std'] / np.sqrt(table_data['cer_count']) * 1.96
        table_data['bwer_ci'] = table_data['bwer_std'] / np.sqrt(table_data['cer_count']) * 1.96

        # Format for display
        for metric in ['cer', 'wer', 'bwer']:
            table_data[f'{metric}_display'] = table_data.apply(
                lambda x: f"{x[f'{metric}_mean']:.4f} ± {x[f'{metric}_ci']:.4f}", axis=1
            )

        # Select and rename columns for display
        display_table = table_data[[
            'provider', 'method', 'model', 'cer_count',
            'cer_display', 'wer_display', 'bwer_display'
        ]].rename(columns={
            'cer_count': 'Samples',
            'cer_display': 'CER (mean ± 95% CI)',
            'wer_display': 'WER (mean ± 95% CI)',
            'bwer_display': 'BWER (mean ± 95% CI)'
        })

        # Sort by WER (typically the primary metric)
        display_table = display_table.sort_values('method')

        comparison_table = mo.vstack([
            mo.md("#### Comprehensive Comparison Table"),
            mo.ui.table(display_table)
        ])

        # Combine all visualizations
        return mo.vstack([
            mo.md("#### Cross-Method and Cross-Provider Comparison"),
            mo.md("This visualization compares performance across all methods and providers."),
            mo.md("#### Method-wise Comparison"),
            *method_wise_chart,
            mo.md("#### Error Rates by Provider and Method"),
            *metric_grid,
            mo.md("#### Box Plots with Outliers"),
            *boxplot_charts,
            mo.md("#### Error Type Breakdown"),
            *ranking_chart,
            comparison_table
        ])
    return (
        create_cross_method_comparison,
        display_comparison_stats,
        extract_text_from_xml,
        plot_error_type_breakdown,
        plot_method_comparison,
        plot_metric_distribution,
    )


@app.cell
def _(mo):
    mo.md(r"""## 1. Transkribus Baseline Evaluation""")
    return


@app.cell
def evaluate_transkribus_baseline(mo):
    from src.transkribus import evaluate_transkribus

    # Dictionary to store results for each dataset
    transkribus_results = {}

    # Define a function to evaluate and display a dataset
    def evaluate_dataset(name, gt_dir, transkribus_dir, output_dir):
        print(f"Evaluating {name.title()} dataset...")
        df = evaluate_transkribus(
            ground_truth_dir=gt_dir,
            transkribus_dir=transkribus_dir,
            output_dir=output_dir,
            save_transcriptions=False
        )
        transkribus_results[name] = df

        # Display results
        if not df.empty:
            avg_cer = df['cer'].mean()
            avg_wer = df['wer'].mean()
            avg_bwer = df['bwer'].mean()
            doc_count = len(df)

            mo.output.append(mo.md(f"### {name.title()} Dataset - Transkribus Text Titan 1 Results"))
            mo.output.append(mo.vstack([
                mo.hstack([
                    mo.stat(f"{avg_cer:.4f}", label="Average CER", bordered=True),
                    mo.stat(f"{avg_wer:.4f}", label="Average WER", bordered=True),
                    mo.stat(f"{avg_bwer:.4f}", label="Average BWER", bordered=True),
                    mo.stat(doc_count, label="Documents", bordered=True)
                ])
            ]))

            mo.output.append(mo.md(f"### {name.title()} - Detailed Results by Document"))
            mo.output.append(mo.callout(mo.plain(df)))
        else:
            mo.output.append(mo.md(f"## ⚠️ No Transkribus results available for {name.title()} dataset"))

    # Evaluate Reichenau dataset
    evaluate_dataset(
        "reichenau", 
        'data/reichenau_10_test/ground_truth',
        'results/linear_transcription/reichenau_inkunabeln/transkribus_10_test',
        'temp'
    )

    # Evaluate Bentham dataset
    evaluate_dataset(
        "bentham", 
        'data/bentham_10_test/ground_truth_renamed',
        'results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
        'bentham_temp'
    )
    return evaluate_dataset, evaluate_transkribus, transkribus_results


@app.cell
def _(mo):
    mo.md(r"""## 2. Cross-Method Comparison""")
    return


@app.cell
def compare_methods(
    create_cross_method_comparison,
    json,
    mo,
    os,
    pd,
    transkribus_results,
    variant_note,
):
    # Function to load results for a specific dataset and evaluation type
    def load_dataset_results(dataset_name, eval_type, method_type="pagewise"):
        """Load results for a specific evaluation type from temp folder"""
        # Define the possible directories to check based on dataset name and method type
        if dataset_name == "bentham":
            if method_type == "pagewise":
                directories = [{"prefix": "bentham_", "temp_dir": "temp"}, {"prefix": "", "temp_dir": "bentham_temp"}]
                eval_suffix = ""
            else:  # linewise
                directories = [{"prefix": "", "temp_dir": "bentham_temp_lines"}]
                eval_suffix = ""
        else:  # reichenau
            if method_type == "pagewise":
                directories = [{"prefix": "", "temp_dir": "temp"}]
                eval_suffix = ""
            else:  # linewise
                directories = [{"prefix": "", "temp_dir": "reichenau_temp_lines"}]
                eval_suffix = ""

        # Adjust evaluation type with suffix for linewise methods
        eval_type_with_suffix = f"{eval_type}{eval_suffix}"

        # Special case for hybrid methods with lines suffix
        if method_type == "linewise" and "hybrid" in eval_type:
            if eval_type == "one_shot_hybrid_enhanced":
                potential_names = ["one_shot_hybrid_lines", "one_shot_hybrid_gt_lines", "one_shot_hybrid_simplified"]
            elif eval_type == "two_shot_hybrid_enhanced":
                potential_names = ["two_shot_hybrid_lines"]
            else:  # hybrid_zero_shot
                potential_names = ["hybrid_zero_shot_lines"]
        else:
            # For linewise methods, add _lines suffix to the eval_type
            if method_type == "linewise" and eval_suffix == "":
                potential_names = [f"{eval_type_with_suffix}_lines"]
            else:
                potential_names = [eval_type_with_suffix]

        # Check all possible directories and potential filenames
        for dir_config in directories:
            prefix = dir_config["prefix"]
            temp_dir = dir_config["temp_dir"]

            for potential_name in potential_names:
                comparison_path = f"{prefix}{temp_dir}/{potential_name}_comparison.json"

                # For debugging, print the path we're checking
                print(f"Checking path: {comparison_path}")

                if os.path.exists(comparison_path):
                    print(f"Found {eval_type} results for {dataset_name} ({method_type}) in {comparison_path}")

                    # Load comparison data
                    with open(comparison_path, 'r') as f:
                        comparison_data = json.load(f)

                    # Print providers found in comparison data
                    print(f"  Providers found: {', '.join(comparison_data.keys())}")

                    # Load provider results
                    provider_results = {}
                    for provider in comparison_data.keys():
                        if provider == "transkribus":
                            continue

                        provider_results_path = f"{prefix}{temp_dir}/{potential_name}/{provider}/{provider}_all_results.csv"
                        if os.path.exists(provider_results_path):
                            provider_results[provider] = pd.read_csv(provider_results_path)
                            print(f"  Loaded results for {provider} from {provider_results_path}")
                        else:
                            print(f"  WARNING: Could not find results for {provider} at {provider_results_path}")

                    # If we found results, store them and add source info
                    if provider_results:
                        results = {
                            "comparison_data": comparison_data,
                            "provider_results": provider_results,
                            "source": temp_dir,
                            "method_type": method_type,
                            "eval_variant": potential_name
                        }
                        return results
                    else:
                        print(f"  WARNING: Found comparison data but no provider results in {comparison_path}")

        # If we didn't find results in any directory, return None
        print(f"No results found for {dataset_name} {eval_type} ({method_type})")
        return None

    # Define evaluation types to load
    eval_types = [
        "zero_shot",
        "one_shot",
        "two_shot",
        "hybrid_zero_shot",
        "one_shot_hybrid_enhanced",  # Note the naming difference
        "two_shot_hybrid_enhanced"   # Note the naming difference
    ]

    # Function to create cross-method comparison with scaled images
    def create_scaled_comparison(ds_name, method_type, results_dict, transkribus_df):
        """Create comparison visualization with properly scaled images"""
        if not results_dict:
            return mo.md(f"⚠️ No {method_type} transcription results found for {ds_name}")

        # Add method_type to the dataset for better labeling in visualizations
        for method_name, result in results_dict.items():
            if "provider_results" in result:
                for provider, df in result["provider_results"].items():
                    if not df.empty and "method_type" not in df.columns:
                        df["method_type"] = method_type

        # # Create method-specific notes
        # notes = []
        # if method_type == "linewise":
        #     for method, result in results_dict.items():
        #         if "eval_variant" in result:
        #             notes.append(f"- {method}: using variant '{result['eval_variant']}'")

        #     variant_note = ""
        #     if notes:
        #         variant_note = mo.md("**Linewise variants used:**\n" + "\n".join(notes))

        # Get the visualization content from the comparison function
        content = create_cross_method_comparison(results_dict, transkribus_df, mo, pd)

        # Find and modify any markdown content with embedded images to include width styling
        modified_content = []

        # Process each element in the content
        for item in content.items if hasattr(content, 'items') else [content]:
            if hasattr(item, '_repr_markdown_') and '<img src="data:image/png;base64,' in item._repr_markdown_():
                # Replace the width=100% with width=100% style="max-width:100%"
                markdown = item._repr_markdown_()
                modified_markdown = markdown.replace('width="100%"', 'width="100%" style="max-width:100%; height:auto;"')
                modified_content.append(mo.md(modified_markdown))
            else:
                modified_content.append(item)

        # Add variant note for linewise method if available
        if method_type == "linewise" and 'variant_note' in locals():
            return mo.vstack([
                variant_note,
                *modified_content
            ])
        else:
            return mo.vstack(modified_content)

    # Process each dataset and return its visualizations
    def process_dataset(ds_name):
        # Get the Transkribus dataframe for this dataset
        current_transkribus_df = transkribus_results[ds_name]

        # Load results for each evaluation type and method type (pagewise vs linewise)
        pagewise_results = {}
        linewise_results = {}

        for current_eval_type in eval_types:
            # Try to load pagewise results
            pagewise_method_results = load_dataset_results(ds_name, current_eval_type, "pagewise")
            if pagewise_method_results:
                method_key = current_eval_type
                if current_eval_type == "one_shot_hybrid_enhanced":
                    method_key = "one_shot_hybrid"
                elif current_eval_type == "two_shot_hybrid_enhanced":
                    method_key = "two_shot_hybrid"
                pagewise_results[method_key] = pagewise_method_results

            # Try to load linewise results
            linewise_method_results = load_dataset_results(ds_name, current_eval_type, "linewise")
            if linewise_method_results:
                method_key = current_eval_type
                if current_eval_type == "one_shot_hybrid_enhanced":
                    method_key = "one_shot_hybrid"
                elif current_eval_type == "two_shot_hybrid_enhanced":
                    method_key = "two_shot_hybrid"
                linewise_results[method_key] = linewise_method_results

        # Log what we found with more details
        print(f"{ds_name.title()}: Found {len(pagewise_results)} pagewise methods and {len(linewise_results)} linewise methods")
        if pagewise_results:
            print(f"  Pagewise methods: {', '.join(pagewise_results.keys())}")
        if linewise_results:
            print(f"  Linewise methods: {', '.join(linewise_results.keys())}")
            for method, result in linewise_results.items():
                if "eval_variant" in result:
                    print(f"    {method}: variant={result['eval_variant']}")

        # Create scaled visualizations
        pagewise_viz = create_scaled_comparison(ds_name, "pagewise", pagewise_results, current_transkribus_df)
        linewise_viz = create_scaled_comparison(ds_name, "linewise", linewise_results, current_transkribus_df)

        # Create a tabbed interface for this dataset
        dataset_tabs = mo.ui.tabs({
            "Pagewise": pagewise_viz,
            "Linewise": linewise_viz
        })

        return dataset_tabs

    # Create visualizations for each dataset using tabs
    reichenau_tabs = process_dataset("reichenau")
    bentham_tabs = process_dataset("bentham")

    # Create a vertical layout with both datasets
    mo.output.append(mo.vstack([
        mo.md("# Reichenau Dataset Results"),
        reichenau_tabs,
        mo.md("# Bentham Dataset Results"),
        bentham_tabs
    ]))
    return (
        bentham_tabs,
        create_scaled_comparison,
        eval_types,
        load_dataset_results,
        process_dataset,
        reichenau_tabs,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
