import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    import sys
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return mo, os, project_root, sys


@app.cell(hide_code=True)
def display_comparison_stats(os):
    from src.visualization import plot_method_comparison, plot_error_type_breakdown, plot_metric_distribution
    from src.file_utils import extract_text_from_xml
    def display_comparison_stats(results, transkribus_df, mo, pd, provider_suffix=""):
        """
        Display comparison stats between models and Transkribus with visualizations

        Args:|
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
                final_provider_stats.append(mo.md(f"### {display_name} with {_model_name}"))
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
                final_provider_stats.append(mo.md(f"### {display_name} with {_model_name}"))
                final_provider_stats.append(
                    mo.hstack([
                        mo.stat(f"{metrics['avg_cer']:.4f}", label="Average CER", bordered=True),
                        mo.stat(f"{metrics['avg_wer']:.4f}", label="Average WER", bordered=True),
                        mo.stat(f"{metrics['avg_bwer']:.4f}", label="Average BWER", bordered=True),
                        mo.stat(metrics["doc_count"], label="Documents", bordered=True)
                    ])
                )

        # Combine everything into a vstack layout for stats
        stats_ui = mo.vstack([
            mo.md("## Model Performance vs Transkribus Baseline"),
            *transkribus_stats,
            *final_provider_stats
        ])

        # Prepare data for visualizations
        all_results = []

        # Convert API results to a unified dataframe for plotting
        for provider, provider_df in results["provider_results"].items():
            display_name = f"{provider}{provider_suffix}"
            for _, doc_metrics in provider_df.iterrows():
                result_row = {
                    'document_id': doc_metrics['document_id'],
                    'method': display_name,
                    'cer': doc_metrics['cer'],
                    'wer': doc_metrics['wer'],
                    'bwer': doc_metrics['bwer']
                }
                all_results.append(result_row)

        # Add Transkribus results if available
        if not transkribus_df.empty:
            for _, row in transkribus_df.iterrows():
                result_row = {
                    'document_id': row['document_id'],
                    'method': 'Transkribus',
                    'cer': row['cer'],
                    'wer': row['wer'],
                    'bwer': row['bwer']
                }
                all_results.append(result_row)

        # Create visualization components
        viz_components = []

        if all_results:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import base64
            import io
            plt.switch_backend('Agg')  # Use non-interactive backend

            results_df = pd.DataFrame(all_results)

            # Only create visualizations if we have at least 2 methods to compare
            if len(results_df['method'].unique()) >= 2:
                # Method comparison visualization
                viz_components.append(mo.md("## Visualizations"))

                # Implement a simplified version of plot_method_comparison that works with a DataFrame
                metrics = ['cer', 'wer', 'bwer']
                plt.figure(figsize=(14, 8))

                # Calculate aggregate metrics for each method
                agg_df = results_df.groupby('method')[metrics].mean().reset_index()

                # Plot each metric
                for i, metric in enumerate(metrics):
                    plt.subplot(1, len(metrics), i+1)

                    # Create bar plot
                    ax = sns.barplot(x='method', y=metric, data=agg_df)

                    # Add value labels on top of bars
                    for j, v in enumerate(agg_df[metric]):
                        ax.text(j, v + 0.01, f"{v:.3f}", ha='center')

                    # Set labels
                    plt.title(f'{metric.upper()}')
                    plt.ylabel(metric.upper())
                    plt.xlabel('Method')

                    # Rotate x-axis labels
                    plt.xticks(rotation=45, ha='right')

                plt.suptitle('Comparison of Methods', fontsize=16)
                plt.tight_layout()

                # Convert figure to base64 for Marimo display
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                viz_components.append(mo.md("### Performance Comparison Across Methods"))
                viz_components.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Method Comparison" />'))

                # Implement a simplified version of plot_error_type_breakdown
                plt.figure(figsize=(12, 8))

                # Calculate the two error components
                error_df = results_df.copy()
                error_df['content_errors'] = error_df['bwer']
                error_df['order_errors'] = error_df['wer'] - error_df['bwer']

                # Reshape for stacked bar chart
                plot_df = pd.melt(error_df,
                               id_vars=['method'],
                               value_vars=['content_errors', 'order_errors'],
                               var_name='Error Type', value_name='Error Rate')

                # Group by method and error type to get averages
                plot_df = plot_df.groupby(['method', 'Error Type'])['Error Rate'].mean().reset_index()

                # Create stacked bar chart
                sns.barplot(x='method', y='Error Rate', hue='Error Type', data=plot_df)

                plt.title('Breakdown of Content vs. Reading Order Errors')
                plt.xlabel('Method')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Error Rate')
                plt.legend(title='Error Type')
                plt.tight_layout()

                # Convert figure to base64 for Marimo display
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                viz_components.append(mo.md("### Content vs Reading Order Errors"))
                viz_components.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Error Breakdown" />'))

                # Create error distributions
                viz_components.append(mo.md("### Error Distributions"))
                dist_html = []

                # Simplified version of plot_metric_distribution
                metrics_to_plot = ['cer', 'wer', 'bwer']
                for metric in metrics_to_plot:
                    plt.figure(figsize=(10, 6))
                    sns.set(style="whitegrid")

                    # Create distribution plot
                    ax = sns.boxplot(x='method', y=metric, data=results_df)
                    sns.stripplot(x='method', y=metric, data=results_df,
                              size=4, color=".3", alpha=0.7)

                    # Rotate x-axis labels
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # Set labels
                    plt.ylabel(metric.upper())
                    plt.title(f'Distribution of {metric.upper()} by Method')

                    # Convert figure to base64 for Marimo display
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                    dist_html.append(f'<img src="data:image/png;base64,{img_str}" alt="{metric.upper()} Distribution" style="width:50%" />')

                if dist_html:
                    viz_components.append(mo.md(f'<div style="display:flex;flex-direction:row">{" ".join(dist_html)}</div>'))

        # Create comparison table
        comparison_df = None
        table_ui = None

        if transkribus_metrics["avg_cer"] is not None:
            # Add Transkribus to comparison data
            comparison_with_baseline = results["comparison_data"].copy()
            comparison_with_baseline["transkribus"] = {
                "model": "Text Titan 1",
                "avg_cer": transkribus_metrics["avg_cer"],
                "avg_wer": transkribus_metrics["avg_wer"],
                "avg_bwer": transkribus_metrics["avg_bwer"],
                "doc_count": len(transkribus_df)
            }

            # Create comparison table
            comparison_df = pd.DataFrame(comparison_with_baseline).T
            table_ui = mo.vstack([
                mo.md("## Comparison Table (All Models)"),
                mo.ui.table(comparison_df.reset_index().rename(columns={"index": "provider"}))
            ])

        # Return both the complete UI and comparison dataframe
        return mo.vstack([
            stats_ui,
            table_ui or mo.md(""),
            *viz_components
        ]), comparison_df




    ###################

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
        import matplotlib.pyplot as plt
        import seaborn as sns
        import base64
        import io
        import numpy as np
        plt.switch_backend('Agg')  # Use non-interactive backend

        # Prepare unified dataframe for all results
        all_method_results = []

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
            if eval_type == "zero_shot_lines":
                display_method = "Zero Shot Linewise"
            elif eval_type == "hybrid_zero_shot_lines":
                display_method = "Hybrid Zero Shot Linewise"
            elif eval_type == "one_shot_lines":
                display_method = "One Shot Linewise"
            elif eval_type == "one_shot_hybrid_lines":
                display_method = "One Shot Hybrid Linewise"
            elif eval_type == "two_shot_lines":
                display_method = "Two Shot Linewise"
            elif eval_type == "two_shot_hybrid_lines":
                display_method = "Two Shot Hybrid Linewise"
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

        # Create a figure for method-wise comparison with dynamic sizing
        width = max(24, 6 * len(method_groups))  # Minimum width of 24 inches
        height = 12

        plt.figure(figsize=(width, height))

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

            # Explicitly create a grouped bar plot directly using the data
            agg_df = unified_df.groupby(['method', 'provider'], observed=True)[metric].mean().reset_index()

            # Find baseline value
            baseline_data = agg_df[(agg_df['method'] == 'Baseline') & (agg_df['provider'] == 'Transkribus')]
            if not baseline_data.empty:
                baseline_value = baseline_data[metric].values[0]
                baseline_values[metric] = baseline_value
            else:
                baseline_value = None

            # Create grouped bar plot using seaborn
            ax = sns.barplot(x='method', y=metric, hue='provider', data=agg_df,
                            palette=provider_colors, errorbar=('ci', 95))

            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)

            # Add reference line for baseline value if it exists
            if baseline_value is not None:
                plt.axhline(y=baseline_value, color='orange', linestyle=':', linewidth=1.5,
                           alpha=0.8, label=f'Baseline ({baseline_value:.3f})')

            # Set axis labels and title
            plt.ylabel(metric_titles[metric], fontsize=12)
            if mi == len(metrics) - 1:  # Only for the last subplot
                plt.xlabel('Method', fontsize=12)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=30, ha='right')

            # Add grid for readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Create proper legend with baseline included
            handles, labels = ax.get_legend_handles_labels()
            if baseline_value is not None:
                baseline_line = plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=1.5)
                handles.append(baseline_line)
                labels.append(f'Baseline ({baseline_value:.3f})')

            plt.legend(handles=handles, labels=labels, title='Provider', loc='best')

        plt.suptitle('Method-wise Comparison by Provider', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust to give more space for suptitle

        # Save figure with bbox_inches='tight' to ensure all elements fit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        method_wise_chart.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Method-wise Comparison" width="100%" />'))

        # Create a grid of metric visualizations
        metric_grid = []

        # Create a figure for all metric combinations
        plt.figure(figsize=(18, 18))  # Increased from (15, 15)

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

        # Save figure with bbox_inches='tight'
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        metric_grid.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Performance Grid" width="100%" />'))

        # 3. Create overall ranking chart with Transkribus in grey
        ranking_chart = []

        # Calculate means for each provider-method combination
        avg_metrics = unified_df.groupby('provider_method')[['cer', 'wer', 'bwer']].mean().reset_index()

        plt.figure(figsize=(18, 12))  # Increased from (15, 10)
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

        # Save figure with bbox_inches='tight'
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        ranking_chart.append(mo.md(f'<img src="data:image/png;base64,{img_str}" alt="Ranking Chart" width="100%" />'))

        # 5. Create comprehensive comparison table with confidence intervals
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
            mo.md("### Comprehensive Comparison Table"),
            mo.ui.table(display_table)
        ])

        # Combine all visualizations
        return mo.vstack([
            mo.md("## Cross-Method and Cross-Provider Comparison"),
            mo.md("This visualization compares performance across all methods and providers."),
            mo.md("### Method-wise Comparison"),
            *method_wise_chart,
            mo.md("### Error Rates by Provider and Method"),
            *metric_grid,
            mo.md("### Error Type Breakdown"),
            *ranking_chart,
            comparison_table
        ])
    ##################

    def get_example_content(num_examples=1):
        """
        Get content of example page(s) for few-shot learning

        Args:
            num_examples: Number of examples to retrieve (1 for one-shot, 2 for two-shot)

        Returns:
            Dictionary with example content and success status
        """
        examples = []

        example_paths = [
            {
                "gt_path": 'data/bentham_10_test/few-shot-samples/116649001.xml',
                "img_path": 'data/bentham_10_test/few-shot-samples/116649001.jpg'
            },
            {
                "gt_path": 'data/bentham_10_test/few-shot-samples/116643002.xml',
                "img_path": 'data/bentham_10_test/few-shot-samples/116643002.jpg'
            }
        ]

        example_paths = example_paths[:num_examples]

        for i, paths in enumerate(example_paths):
            gt_path = paths["gt_path"]
            img_path = paths["img_path"]

            # Check if files exist
            if os.path.exists(gt_path) and os.path.exists(img_path):
                # Extract text from ground truth
                lines = extract_text_from_xml(gt_path)
                text = "\n".join(lines) if lines else ""

                if text:
                    examples.append({
                        "text": text,
                        "image_path": img_path,
                        "success": True
                    })
                else:
                    examples.append({"success": False})
            else:
                examples.append({"success": False})

        # Check overall success
        all_success = all(example["success"] for example in examples)

        # Return according to number of examples requested
        if num_examples == 1 and examples:
            return examples[0]
        else:
            return {
                "examples": examples,
                "success": all_success
            }

        return get_example_content
    return (
        create_cross_method_comparison,
        display_comparison_stats,
        extract_text_from_xml,
        get_example_content,
        plot_error_type_breakdown,
        plot_method_comparison,
        plot_metric_distribution,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Phase 1: LINEWISE Quantitative Transkription - [Bentham Papers]()

        Handwritten English Pages
        ---
        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Traditional OCR Model - [Transkribus Text Titan 1](https://app.transkribus.org/models/text/51170)""")
    return


@app.cell
def _(mo):
    from src.transkribus import evaluate_transkribus
    import pandas as pd
    transkribus_df = evaluate_transkribus(
        ground_truth_dir='data/bentham_10_test/ground_truth_renamed',
        transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
        output_dir='bentham_temp_lines',
        save_transcriptions=False
    )

    if not transkribus_df.empty:
        transkribus_avg_cer = transkribus_df['cer'].mean()
        transkribus_avg_wer = transkribus_df['wer'].mean()
        transkribus_avg_bwer = transkribus_df['bwer'].mean()
        transkribus_doc_count = len(transkribus_df)

        mo.output.append(mo.md("### Transkribus Text Titan 1 Results"))
        mo.output.append(mo.vstack([
            mo.hstack([
                mo.stat(f"{transkribus_avg_cer:.4f}", label="Average CER", bordered=True),
                mo.stat(f"{transkribus_avg_wer:.4f}", label="Average WER", bordered=True),
                mo.stat(f"{transkribus_avg_bwer:.4f}", label="Average BWER", bordered=True),
                mo.stat(transkribus_doc_count, label="Documents", bordered=True)
            ])
        ]))

        mo.output.append(mo.md("### Detailed Results by Document"))
        mo.output.append(mo.callout(mo.plain(transkribus_df)))
    else:
        mo.md("## ⚠️ No Transkribus results available")
    return (
        evaluate_transkribus,
        pd,
        transkribus_avg_bwer,
        transkribus_avg_cer,
        transkribus_avg_wer,
        transkribus_df,
        transkribus_doc_count,
    )


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Multi Modal Large Language Models Evaluation""")
    return


@app.cell
def _(mo):
    # siehe https://www.blb-karlsruhe.de/blblog/2023-10-13-projekt-reichenauer-inkunabeln Meta Infos zum Buch

    system_prompt = mo.ui.text_area(label="System Prompt", full_width=True, rows=20, value="""
    You are a expert for historical english handwritten manuscripts — especially those authored by Jeremy Bentham (1748–1832).
    Please transcribe the provided line image exactly as it appears. Transcribe exactly what you see in the image, preserving the original text without modernizing or correcting spelling.

    Important instructions:
    1. Use original historical characters and spelling.
    2. Preserve abbreviations, marginalia, and special characters.
    4. Do not add any explanations or commentary
    5. Do not include line numbers
    6. Transcribe text only, ignore decorative elements or stamps unless they contain readable text
    7. Ignore text that is clearly struck through in the manuscript

    If you're uncertain about a character, provide your best interpretation.

    """)

    provider_models = {
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "mistral": "mistral-small-latest",

    }

    limit_docs = 10

    system_prompt
    return limit_docs, provider_models, system_prompt


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""### MM-LLM Zero-Shot Evaluation""")
    return


@app.cell
def _(mo):
    from src.evaluation import run_evaluation, get_transkribus_text, run_model_evaluation, process_page_by_lines
    from src.file_utils import encode_image, encode_image_object
    import time

    zero_shot_run_button = mo.ui.run_button(
        label="Run Zero-Shot Evaluation",
        kind="success",
        tooltip="Start the zero-shot evaluation process"
    )



    mo.vstack([
        mo.md("## Zero-Shot Evaluation for MM-LLMs"),
        zero_shot_run_button
    ])
    return (
        encode_image,
        encode_image_object,
        get_transkribus_text,
        process_page_by_lines,
        run_evaluation,
        run_model_evaluation,
        time,
        zero_shot_run_button,
    )


@app.cell
def _(
    encode_image_object,
    limit_docs,
    provider_models,
    run_model_evaluation,
    system_prompt,
    zero_shot_run_button,
):
    if zero_shot_run_button.value:
        def create_zero_shot_line_messages(doc_id, line_id, line_image, line_idx):
            # Encode the line image
            line_image_base64 = encode_image_object(line_image)

            return [
                {"role": "system", "content": system_prompt.value},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{line_image_base64}"}
                        }
                    ]
                }
            ]

        # Run the line-based evaluation with the new syntax
        zero_shot_line_results = run_model_evaluation(
            strategy="line",  # Specify line strategy explicitly
            provider_models=provider_models,
            gt_dir='data/bentham_10_test/ground_truth_renamed',
            image_dir='data/bentham_10_test/images_renamed',
            transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
            base_output_dir='bentham_temp_lines',
            create_messages=create_zero_shot_line_messages,  # Note: parameter renamed from create_line_messages
            eval_type='zero_shot_lines',
            limit=limit_docs,
            parallel=True,
            max_workers=None,
            use_structured_output=True 
        )
    return create_zero_shot_line_messages, zero_shot_line_results


@app.cell
def _(
    display_comparison_stats,
    mo,
    pd,
    transkribus_df,
    zero_shot_line_results,
):
    zero_shot_stats_display, zero_shot_eval = display_comparison_stats(zero_shot_line_results, transkribus_df, mo, pd, provider_suffix="_zero-shot")
    mo.vstack([zero_shot_stats_display, zero_shot_line_results.get("provider_results")])
    return zero_shot_eval, zero_shot_stats_display


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Hybrid Evaluation: MM-LLM Zero Shot + Transkribus""")
    return


@app.cell
def _(mo):
    zero_shot_hybrid_run_button = mo.ui.run_button(
        label="Run Zero-Shot Hybrid Evaluation",
        kind="success",
        tooltip="Start the hybrid evaluation process"
    )

    mo.vstack([
        mo.md("## Zero-Shot Hybrid Evaluation for MM-LLMs"),
        zero_shot_hybrid_run_button
    ])
    return (zero_shot_hybrid_run_button,)


@app.cell
def _(
    encode_image_object,
    limit_docs,
    provider_models,
    run_line_evaluation,
    system_prompt,
    zero_shot_hybrid_run_button,
):
    from src.file_utils import find_file_for_id, extract_line_coords_from_xml

    if zero_shot_hybrid_run_button.value:
        def create_hybrid_line_messages(doc_id, line_id, line_image, line_idx):
            # First, get Transkribus file path for this document
            transkribus_path = find_file_for_id(
                doc_id,
                'results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
                ['.xml']
            )

            # Get the line text from Transkribus for this specific line
            transkribus_line_text = ""
            if transkribus_path:
                # Extract all lines with their IDs from Transkribus XML
                transkribus_lines = extract_line_coords_from_xml(transkribus_path)

                # Find the matching line by index (line_idx should match the position)
                if line_idx < len(transkribus_lines):
                    transkribus_line_text = transkribus_lines[line_idx].get('text', '')

            # Encode the line image
            line_image_base64 = encode_image_object(line_image)

            # Create messages with Transkribus line output as reference
            return [
                {"role": "system", "content": system_prompt.value},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{line_image_base64}"}
                        },
                        {"type": "text", "text": f"The following is the output of a traditional OCR model (Transkribus) for this line. It can help with your transcription, but may contain errors:\n\n\"{transkribus_line_text}\""},
                    ]
                }
            ]

        hybrid_zero_shot_line_results = run_line_evaluation(
            provider_models=provider_models,
            gt_dir='data/bentham_10_test/ground_truth_renamed',
            image_dir='data/bentham_10_test/images_renamed',
            transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
            base_output_dir='bentham_temp_lines',
            create_line_messages=create_hybrid_line_messages,
            eval_type='hybrid_zero_shot_lines',
            limit=limit_docs,
            parallel=True,
            max_workers=None,
            use_structured_output=True 
        )
    return (
        create_hybrid_line_messages,
        extract_line_coords_from_xml,
        find_file_for_id,
        hybrid_zero_shot_line_results,
    )


@app.cell
def _(
    display_comparison_stats,
    hybrid_zero_shot_line_results,
    mo,
    pd,
    transkribus_df,
):
    hybrid_zero_shot_stats_display, hybrid_zero_shot_eval = display_comparison_stats(hybrid_zero_shot_line_results, transkribus_df, mo, pd, provider_suffix="_hybrid_zero-shot")
    mo.vstack([hybrid_zero_shot_stats_display, hybrid_zero_shot_line_results.get("provider_results")])
    return hybrid_zero_shot_eval, hybrid_zero_shot_stats_display


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""### One-Shot Evaluation for MM-LLMs""")
    return


@app.cell
def _(mo):
    one_shot_run_button = mo.ui.run_button(
            label="Run One-Shot Evaluation",
            kind="success",
            tooltip="Start the one-shot evaluation process"
        )

    mo.vstack([
        mo.md("## One-Shot Evaluation for MM-LLMs"),
        one_shot_run_button
    ])
    return (one_shot_run_button,)


@app.cell
def _(
    encode_image_object,
    extract_text_from_xml,
    limit_docs,
    one_shot_run_button,
    os,
    process_page_by_lines,
    provider_models,
    run_line_evaluation,
    system_prompt,
):
    if one_shot_run_button.value:
        # Manually construct the example with the correct paths
        example_img_path = 'data/bentham_10_test/few-shot-samples/116649001.jpg'
        example_xml_path = 'data/bentham_10_test/few-shot-samples/transkribus/0102_116649001.xml'

        # Check if the files exist
        if os.path.exists(example_img_path) and os.path.exists(example_xml_path):
            print("✅ Using examples from dedicated example folder for one-shot learning")

            # Extract text from ground truth
            gt_lines = extract_text_from_xml(example_xml_path)
            gt_text = "\n".join(gt_lines) if gt_lines else ""

            if gt_text:
                # Extract lines from the example page
                os1_line_data = process_page_by_lines(example_img_path, example_xml_path)

                # Use the first line as our example
                if os1_line_data['lines'] and len(os1_line_data['lines']) > 0:
                    os1_line = os1_line_data['lines'][8] # index 8!
                    os1_line_image = os1_line['image']
                    os1_line_text = os1_line['text']

                    # Encode the example line image
                    os1_line_image_base64 = encode_image_object(os1_line_image)

                    def create_one_shot_line_messages(doc_id, line_id, line_image, line_idx):
                        # Encode the target line image
                        line_image_base64 = encode_image_object(line_image)

                        # Create messages with example line and target line
                        return [
                            {
                                "role": "system",
                                "content": system_prompt.value
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Review the following example of a line from a historical handwritten document along with its correct transcription. This shows how to handle abbreviations and typographic features.\n\n**Use this example to learn how to transcribe historical documents — do not copy the example text directly and **"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{os1_line_image_base64}"}
                                    },
                                    {
                                        "type": "text",
                                        "text": f"CORRECT TRANSCRIPTION\n{os1_line_text}"
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Now transcribe the following line:"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{line_image_base64}"}
                                    }
                                ]
                            }
                        ]

                    one_shot_line_results = run_line_evaluation(
                        provider_models=provider_models,
                        gt_dir='data/bentham_10_test/ground_truth_renamed',
                        image_dir='data/bentham_10_test/images_renamed',
                        transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
                        base_output_dir='bentham_temp_lines',
                        create_line_messages=create_one_shot_line_messages,
                        eval_type='one_shot_lines',
                        limit=limit_docs,
                        parallel=True,
                        max_workers=None,
                        use_structured_output=True 
                    )
                else:
                    print("⚠️ Could not extract lines from the example page")
                    one_shot_line_results = None
            else:
                print("⚠️ Could not extract text from the example XML")
                one_shot_line_results = None
        else:
            print(f"⚠️ Could not find example files:")
            if not os.path.exists(example_img_path):
                print(f" - Missing image: {example_img_path}")
            if not os.path.exists(example_xml_path):
                print(f" - Missing XML: {example_xml_path}")
            one_shot_line_results = None
    return (
        create_one_shot_line_messages,
        example_img_path,
        example_xml_path,
        gt_lines,
        gt_text,
        one_shot_line_results,
        os1_line,
        os1_line_data,
        os1_line_image,
        os1_line_image_base64,
        os1_line_text,
    )


@app.cell
def _(display_comparison_stats, mo, one_shot_line_results, pd, transkribus_df):
    one_shot_stats_display, one_shot_eval = display_comparison_stats(
                one_shot_line_results,
                transkribus_df,
                mo,
                pd,
                provider_suffix="_one_shot"
            )
    mo.vstack([one_shot_stats_display, one_shot_line_results.get("provider_results")])
    return one_shot_eval, one_shot_stats_display


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Hybrid Evaluation: MM-LLM One-Shot + Transkribus""")
    return


@app.cell
def _(mo):
    one_shot_hybrid_run_button = mo.ui.run_button(
            label="Run One-Shot Hybrid Evaluation",
            kind="success",
            tooltip="Start the one-shot hybrid evaluation process"
        )

    mo.vstack([
        mo.md("## One-Shot Hybrid Evaluation for MM-LLMs"),
        one_shot_hybrid_run_button
    ])
    return (one_shot_hybrid_run_button,)


@app.cell
def _(
    encode_image_object,
    example_img_path,
    example_xml_path,
    extract_line_coords_from_xml,
    find_file_for_id,
    limit_docs,
    one_shot_hybrid_run_button,
    os,
    process_page_by_lines,
    provider_models,
    run_line_evaluation,
    system_prompt,
):
    if one_shot_hybrid_run_button.value:
        # Manually construct the example with the correct paths
        # example_img_path = 'data/reichenau_10_test/few-shot-samples/7474192.jpg'
        # example_xml_path = 'data/reichenau_10_test/few-shot-samples/transkribus/0001_7474192.xml'
        example_doc_id = '7474192'  # Extract from filename

        # Check if the files exist
        if os.path.exists(example_img_path) and os.path.exists(example_xml_path):
            print("✅ Using examples from dedicated example folder for one-shot hybrid learning")

            # Process the example page to extract line images
            osh_line_data = process_page_by_lines(example_img_path, example_xml_path)

            # Use the first line as our example
            if osh_line_data['lines'] and len(osh_line_data['lines']) > 0:
                osh_line = osh_line_data['lines'][8]
                osh_line_image = osh_line['image']
                osh_line_text = osh_line['text']

                # Get Transkribus text for this line - directly from the example XML
                osh_transkribus_lines = extract_line_coords_from_xml(example_xml_path)
                osh_line_transkribus_text = ""
                if osh_transkribus_lines and len(osh_transkribus_lines) > 0:
                    osh_line_transkribus_text = osh_transkribus_lines[8].get('text', '')

                if not osh_line_transkribus_text:
                    osh_line_transkribus_text = "[No Transkribus transcription available for this example]"
                    print(f"⚠️ No Transkribus transcription found for example line")

                # Encode the example line image
                osh_line_image_base64 = encode_image_object(osh_line_image)

                def create_one_shot_hybrid_line_messages(doc_id, line_id, line_image, line_idx):
                    # Find Transkribus text for this specific line
                    transkribus_path = find_file_for_id(
                        doc_id,
                        'results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
                        ['.xml']
                    )

                    transkribus_line_text = ""
                    if transkribus_path:
                        transkribus_lines = extract_line_coords_from_xml(transkribus_path)
                        if line_idx < len(transkribus_lines):
                            transkribus_line_text = transkribus_lines[line_idx].get('text', '')

                    # Encode target line image
                    line_image_base64 = encode_image_object(line_image)

                    # Create messages with the original structure
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt.value
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Review the following example of historical handwritten line along with the transcriptions. It includes both:\n\n(1) A raw OCR transcription from Transkribus\n(2) The correct ground truth transcription\n\nThis shows how to improve upon the OCR output and handle abbreviations or typographic features.\n\n**Use this example to learn how to improve OCR transcriptions — do not copy the example text directly.**"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{osh_line_image_base64}"}
                                },
                                {
                                    "type": "text",
                                    "text": f"(1) TRANSKRIBUS OCR OUTPUT\n{osh_line_transkribus_text}\n---\n(2) CORRECT TRANSCRIPTION ====\n{osh_line_text}"
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Now transcribe the following line:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{line_image_base64}"}
                                },
                                {
                                    "type": "text",
                                    "text": f"The following is the output of a traditional OCR model from Transkribus for this line. It can help you transcribe this line, but may also contain errors:\n\n{transkribus_line_text}"
                                }
                            ]
                        }
                    ]

                    return messages

                one_shot_hybrid_line_results = run_line_evaluation(
                    provider_models=provider_models,
                    gt_dir='data/bentham_10_test/ground_truth_renamed',
                    image_dir='data/bentham_10_test/images_renamed',
                    transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
                    base_output_dir='bentham_temp_lines',
                    create_line_messages=create_one_shot_hybrid_line_messages,
                    eval_type='one_shot_hybrid_lines',
                    limit=limit_docs,
                    parallel=True,
                    max_workers=None,
                    use_structured_output=True 
                )
            else:
                print("⚠️ Could not extract lines from the example page")
                one_shot_hybrid_line_results = None
        else:
            print(f"⚠️ Could not find example files:")
            if not os.path.exists(example_img_path):
                print(f" - Missing image: {example_img_path}")
            if not os.path.exists(example_xml_path):
                print(f" - Missing XML: {example_xml_path}")
            one_shot_hybrid_line_results = None
    return (
        create_one_shot_hybrid_line_messages,
        example_doc_id,
        one_shot_hybrid_line_results,
        osh_line,
        osh_line_data,
        osh_line_image,
        osh_line_image_base64,
        osh_line_text,
        osh_line_transkribus_text,
        osh_transkribus_lines,
    )


@app.cell
def _(
    display_comparison_stats,
    mo,
    one_shot_hybrid_line_results,
    pd,
    transkribus_df,
):
    one_shot_hybrid_stats_display, one_shot_hybrid_eval = display_comparison_stats(
                one_shot_hybrid_line_results,
                transkribus_df,
                mo,
                pd,
                provider_suffix="_one_shot_hybrid"
            )
    mo.vstack([one_shot_hybrid_stats_display, one_shot_hybrid_line_results.get("provider_results")])
    return one_shot_hybrid_eval, one_shot_hybrid_stats_display


@app.cell
def _(mo):
    mo.md(r"""### Two-Shot Evaluation for MM-LLMs""")
    return


@app.cell
def _(mo):
    two_shot_run_button = mo.ui.run_button(
            label="Run Two-Shot Evaluation",
            kind="success",
            tooltip="Start the two-shot evaluation process"
        )

    mo.vstack([
        mo.md("## Two-Shot Evaluation for MM-LLMs"),
        two_shot_run_button
    ])
    return (two_shot_run_button,)


@app.cell
def _(
    encode_image_object,
    limit_docs,
    os,
    process_page_by_lines,
    provider_models,
    run_line_evaluation,
    system_prompt,
    two_shot_run_button,
):
    if two_shot_run_button.value:
        # Manually construct the examples with the correct paths
        ts_example1_img_path = 'data/bentham_10_test/few-shot-samples/116649001.jpg'
        ts_example1_xml_path = 'data/bentham_10_test/few-shot-samples/transkribus/0102_116649001.xml'

        ts_example2_img_path = 'data/bentham_10_test/few-shot-samples/116643002.jpg'
        ts_example2_xml_path = 'data/bentham_10_test/few-shot-samples/transkribus/0101_116643002.xml'

        # Check if the files exist
        files_exist = os.path.exists(ts_example1_img_path) and os.path.exists(ts_example1_xml_path) and \
                     os.path.exists(ts_example2_img_path) and os.path.exists(ts_example2_xml_path)

        if files_exist:
            print("✅ Using examples from dedicated example folder for two-shot line learning")

            # Process both example pages to extract lines
            ts_example1_line_data = process_page_by_lines(ts_example1_img_path, ts_example1_xml_path)
            ts_example2_line_data = process_page_by_lines(ts_example2_img_path, ts_example2_xml_path)

            # Verify we have lines from both examples
            if (ts_example1_line_data['lines'] and len(ts_example1_line_data['lines']) > 0 and
                ts_example2_line_data['lines'] and len(ts_example2_line_data['lines']) > 0):

                # Get first line from first example
                ts_example1_line = ts_example1_line_data['lines'][8]
                ts_example1_line_image = ts_example1_line['image']
                ts_example1_line_text = ts_example1_line['text']

                # Get first line from second example
                ts_example2_line = ts_example2_line_data['lines'][1]
                ts_example2_line_image = ts_example2_line['image']
                ts_example2_line_text = ts_example2_line['text']

                # Encode line images
                ts_example1_line_image_base64 = encode_image_object(ts_example1_line_image)
                ts_example2_line_image_base64 = encode_image_object(ts_example2_line_image)

                # Create a closure with the examples captured
                def create_message_function(example1_image_base64, example1_text, example2_image_base64, example2_text):
                    def create_two_shot_line_messages(doc_id, line_id, line_image, line_idx):
                        # Encode target line image
                        line_image_base64 = encode_image_object(line_image)

                        # Create messages with both example lines and target line
                        return [
                            {
                                "role": "system",
                                "content": system_prompt.value
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Review the following examples of historical handwritten lines along with their correct transcriptions. This shows how to handle abbreviations and typographic features.\n\n**Use these examples to learn how to transcribe historical documents — do not copy the example text directly.**"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{example1_image_base64}"}
                                    },
                                    {
                                        "type": "text",
                                        "text": f"CORRECT TRANSCRIPTION\n{example1_text}"
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Here's a second example of a historical handwritten line along with its correct transcription:"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{example2_image_base64}"}
                                    },
                                    {
                                        "type": "text",
                                        "text": f"CORRECT TRANSCRIPTION\n{example2_text}"
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Now transcribe the following line:"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{line_image_base64}"}
                                    }
                                ]
                            }
                        ]
                    return create_two_shot_line_messages

                # Create the function with captured examples
                message_creator = create_message_function(
                    ts_example1_line_image_base64,
                    ts_example1_line_text,
                    ts_example2_line_image_base64,
                    ts_example2_line_text
                )

                two_shot_line_results = run_line_evaluation(
                    provider_models=provider_models,
                    gt_dir='data/bentham_10_test/ground_truth_renamed',
                    image_dir='data/bentham_10_test/images_renamed',
                    transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
                    base_output_dir='bentham_temp_lines',
                    create_line_messages=message_creator,
                    eval_type='two_shot_lines',
                    limit=limit_docs,
                    parallel=True,
                    max_workers=None,
                    use_structured_output=True 
                )
            else:
                print("⚠️ Could not extract lines from one or both example pages")
                two_shot_line_results = None
        else:
            print(f"⚠️ Could not find all example files:")
            if not os.path.exists(ts_example1_img_path):
                print(f" - Missing image 1: {ts_example1_img_path}")
            if not os.path.exists(ts_example1_xml_path):
                print(f" - Missing XML 1: {ts_example1_xml_path}")
            if not os.path.exists(ts_example2_img_path):
                print(f" - Missing image 2: {ts_example2_img_path}")
            if not os.path.exists(ts_example2_xml_path):
                print(f" - Missing XML 2: {ts_example2_xml_path}")
            two_shot_line_results = None
    return (
        create_message_function,
        files_exist,
        message_creator,
        ts_example1_img_path,
        ts_example1_line,
        ts_example1_line_data,
        ts_example1_line_image,
        ts_example1_line_image_base64,
        ts_example1_line_text,
        ts_example1_xml_path,
        ts_example2_img_path,
        ts_example2_line,
        ts_example2_line_data,
        ts_example2_line_image,
        ts_example2_line_image_base64,
        ts_example2_line_text,
        ts_example2_xml_path,
        two_shot_line_results,
    )


@app.cell
def _(display_comparison_stats, mo, pd, transkribus_df, two_shot_line_results):
    two_shot_stats_display, two_shot_eval = display_comparison_stats(
                two_shot_line_results,
                transkribus_df,
                mo,
                pd,
                provider_suffix="_two_shot"
            )
    mo.vstack([two_shot_stats_display, two_shot_line_results.get("provider_results")])
    return two_shot_eval, two_shot_stats_display


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Hybrid Evaluation: MM-LLM Two-Shot + Transkribus""")
    return


@app.cell
def _(mo):
    two_shot_hybrid_run_button = mo.ui.run_button(
            label="Run Two-Shot Hybrid Evaluation",
            kind="success",
            tooltip="Start the one-shot hybrid evaluation process"
        )

    mo.vstack([
        mo.md("## Two-Shot Hybrid Evaluation for MM-LLMs"),
        two_shot_hybrid_run_button
    ])
    return (two_shot_hybrid_run_button,)


@app.cell
def _(
    encode_image_object,
    extract_line_coords_from_xml,
    files_exist,
    find_file_for_id,
    limit_docs,
    os,
    process_page_by_lines,
    provider_models,
    run_line_evaluation,
    system_prompt,
    two_shot_hybrid_run_button,
):
    if two_shot_hybrid_run_button.value:
        # Manually construct the examples with the correct paths
        tsh_example1_img_path = 'data/bentham_10_test/few-shot-samples/116649001.jpg'
        tsh_example1_xml_path = 'data/bentham_10_test/few-shot-samples/transkribus/0102_116649001.xml'
        tsh_example1_doc_id = '116649001'

        tsh_example2_img_path = 'data/bentham_10_test/few-shot-samples/116643002.jpg'
        tsh_example2_xml_path = 'data/bentham_10_test/few-shot-samples/transkribus/0101_116643002.xml'
        tsh_example2_doc_id = '116643002'

        # Check if the files exist
        # files_exist = os.path.exists(tsh_example1_img_path) and os.path.exists(tsh_example1_xml_path) and \
        #               os.path.exists(tsh_example2_img_path) and os.path.exists(tsh_example2_xml_path)

        if files_exist:
            print("✅ Using examples from dedicated example folder for two-shot hybrid line learning")

            # Process both example pages to extract lines
            tsh_example1_line_data = process_page_by_lines(tsh_example1_img_path, tsh_example1_xml_path)
            tsh_example2_line_data = process_page_by_lines(tsh_example2_img_path, tsh_example2_xml_path)

            # Verify we have lines from both examples
            if (tsh_example1_line_data['lines'] and len(tsh_example1_line_data['lines']) > 0 and
                tsh_example2_line_data['lines'] and len(tsh_example2_line_data['lines']) > 0):

                # Get first line from first example
                tsh_example1_line = tsh_example1_line_data['lines'][8]
                tsh_example1_line_image = tsh_example1_line['image']
                tsh_example1_line_text = tsh_example1_line['text']

                # Get first line from second example
                tsh_example2_line = tsh_example2_line_data['lines'][1]
                tsh_example2_line_image = tsh_example2_line['image']
                tsh_example2_line_text = tsh_example2_line['text']

                # Get Transkribus text for first example line
                tsh_example1_transkribus_lines = extract_line_coords_from_xml(tsh_example1_xml_path)
                tsh_example1_line_transkribus_text = ""
                if tsh_example1_transkribus_lines and len(tsh_example1_transkribus_lines) > 0:
                    tsh_example1_line_transkribus_text = tsh_example1_transkribus_lines[8].get('text', '')

                if not tsh_example1_line_transkribus_text:
                    tsh_example1_line_transkribus_text = "[No Transkribus transcription available for this example]"
                    print(f"⚠️ No Transkribus transcription found for example 1 line")

                # Get Transkribus text for second example line
                tsh_example2_transkribus_lines = extract_line_coords_from_xml(tsh_example2_xml_path)
                tsh_example2_line_transkribus_text = ""
                if tsh_example2_transkribus_lines and len(tsh_example2_transkribus_lines) > 0:
                    tsh_example2_line_transkribus_text = tsh_example2_transkribus_lines[1].get('text', '')

                if not tsh_example2_line_transkribus_text:
                    tsh_example2_line_transkribus_text = "[No Transkribus transcription available for this example]"
                    print(f"⚠️ No Transkribus transcription found for example 2 line")

                # Encode line images
                tsh_example1_line_image_base64 = encode_image_object(tsh_example1_line_image)
                tsh_example2_line_image_base64 = encode_image_object(tsh_example2_line_image)

                def create_two_shot_hybrid_line_messages(doc_id, line_id, line_image, line_idx):
                    # Find Transkribus text for this specific line
                    transkribus_path = find_file_for_id(
                        doc_id,
                        'results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
                        ['.xml']
                    )

                    transkribus_line_text = ""
                    if transkribus_path:
                        transkribus_lines = extract_line_coords_from_xml(transkribus_path)
                        if line_idx < len(transkribus_lines):
                            transkribus_line_text = transkribus_lines[line_idx].get('text', '')

                    # Encode target line image
                    line_image_base64 = encode_image_object(line_image)

                    # Create messages with both example lines and target line
                    return [
                        {
                            "role": "system",
                            "content": system_prompt.value
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Review the following examples of historical handwritten lines along with the transcriptions. It includes both:\n\n(1) A raw OCR transcription from Transkribus\n(2) The correct ground truth transcription\n\nThis shows how to improve upon the OCR output and handle abbreviations or typographic features.\n\n**Use these examples to learn how to improve OCR transcriptions — do not copy the example text directly.**"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{tsh_example1_line_image_base64}"}
                                },
                                {
                                    "type": "text",
                                    "text": f"(1) TRANSKRIBUS OCR OUTPUT\n{tsh_example1_line_transkribus_text}\n---\n(2) CORRECT TRANSCRIPTION\n{tsh_example1_line_text}"
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Here's a second example:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{tsh_example2_line_image_base64}"}
                                },
                                {
                                    "type": "text",
                                    "text": f"(1) TRANSKRIBUS OCR OUTPUT\n{tsh_example2_line_transkribus_text}\n---\n(2) CORRECT TRANSCRIPTION\n{tsh_example2_line_text}"
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Now transcribe the following line:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{line_image_base64}"}
                                },
                                {
                                    "type": "text",
                                    "text": f"The following is the output of a traditional OCR model from Transkribus for this line. It is fine-tuned on historic handwritten texts. It can help you transcribe this line, but may also contain errors:\n\n{transkribus_line_text}"
                                }
                            ]
                        }
                    ]

                two_shot_hybrid_line_results = run_line_evaluation(
                    provider_models=provider_models,
                    gt_dir='data/bentham_10_test/ground_truth_renamed',
                    image_dir='data/bentham_10_test/images_renamed',
                    transkribus_dir='results/linear_transcription/bentham_papers/transkribus_10_test_renamed',
                    base_output_dir='bentham_temp_lines',
                    create_line_messages=create_two_shot_hybrid_line_messages,
                    eval_type='two_shot_hybrid_lines',
                    limit=limit_docs,
                    parallel=True,
                    max_workers=None,
                    use_structured_output=True 
                )
            else:
                print("⚠️ Could not extract lines from one or both example pages")
                two_shot_hybrid_line_results = None
        else:
            print(f"⚠️ Could not find all example files:")
            if not os.path.exists(tsh_example1_img_path):
                print(f" - Missing image 1: {tsh_example1_img_path}")
            if not os.path.exists(tsh_example1_xml_path):
                print(f" - Missing XML 1: {tsh_example1_xml_path}")
            if not os.path.exists(tsh_example2_img_path):
                print(f" - Missing image 2: {tsh_example2_img_path}")
            if not os.path.exists(tsh_example2_xml_path):
                print(f" - Missing XML 2: {tsh_example2_xml_path}")
            two_shot_hybrid_line_results = None
    return (
        create_two_shot_hybrid_line_messages,
        tsh_example1_doc_id,
        tsh_example1_img_path,
        tsh_example1_line,
        tsh_example1_line_data,
        tsh_example1_line_image,
        tsh_example1_line_image_base64,
        tsh_example1_line_text,
        tsh_example1_line_transkribus_text,
        tsh_example1_transkribus_lines,
        tsh_example1_xml_path,
        tsh_example2_doc_id,
        tsh_example2_img_path,
        tsh_example2_line,
        tsh_example2_line_data,
        tsh_example2_line_image,
        tsh_example2_line_image_base64,
        tsh_example2_line_text,
        tsh_example2_line_transkribus_text,
        tsh_example2_transkribus_lines,
        tsh_example2_xml_path,
        two_shot_hybrid_line_results,
    )


@app.cell
def _(
    display_comparison_stats,
    mo,
    pd,
    transkribus_df,
    two_shot_hybrid_line_results,
):
    two_shot_hybrid_stats_display, two_shot_hybrid_eval = display_comparison_stats(
                two_shot_hybrid_line_results,
                transkribus_df,
                mo,
                pd,
                provider_suffix="_two_shot_hybrid"
            )
    mo.vstack([two_shot_hybrid_stats_display, two_shot_hybrid_line_results.get("provider_results")])
    return two_shot_hybrid_eval, two_shot_hybrid_stats_display


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Comparison across all Methods
        """
    )
    return


@app.cell
def _(create_cross_method_comparison, mo, os, pd, transkribus_df):
    import json

    def load_results_from_temp(eval_type):
        """Load results for a specific evaluation type from temp folder"""
        comparison_path = f"bentham_temp_lines/{eval_type}_comparison.json"
        if not os.path.exists(comparison_path):
            return None

        # Load comparison data
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)

        # Load provider result
        provider_results = {}
        for provider in comparison_data.keys():
            if provider == "transkribus":
                continue

            provider_results_path = f"bentham_temp_lines/{eval_type}/{provider}/{provider}_all_results.csv"
            if os.path.exists(provider_results_path):
                provider_results[provider] = pd.read_csv(provider_results_path)

        return {
            "comparison_data": comparison_data,
            "provider_results": provider_results
        }

    # Define evaluation types to load
    eval_types = [
        "zero_shot_lines",
        "one_shot_lines",
        "two_shot_lines",
        "hybrid_zero_shot_lines",
        "one_shot_hybrid_lines",
        "two_shot_hybrid_lines"
    ]

    # Load results for each evaluation type
    all_methods_results = {}
    for eval_type in eval_types:
        results = load_results_from_temp(eval_type)
        if results and results["provider_results"]:
            # Map the file naming to the variable naming used in your code
            if eval_type == "one_shot_hybrid_lines":
                all_methods_results["one_shot_hybrid_lines"] = results
            elif eval_type == "two_shot_hybrid_lines":
                all_methods_results["two_shot_hybrid_lines"] = results
            else:
                all_methods_results[eval_type] = results

    # Log what we found
    print(f"Found results for {len(all_methods_results)} evaluation methods: {', '.join(all_methods_results.keys())}")

    # Create comprehensive comparison visualization
    if all_methods_results:
        comparison_viz = create_cross_method_comparison(all_methods_results, transkribus_df, mo, pd)
        mo.output.append(comparison_viz)
    else:
        mo.md("No evaluation results found in temp folder. Please run at least one evaluation method.")
    return (
        all_methods_results,
        comparison_viz,
        eval_type,
        eval_types,
        json,
        load_results_from_temp,
        results,
    )


if __name__ == "__main__":
    app.run()
