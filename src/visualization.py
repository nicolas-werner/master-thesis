import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

def plot_metric_distribution(df: pd.DataFrame, metric: str = 'wer',
                           by_method: bool = True, output_path: Optional[str] = None):
    """
    Plot the distribution of a metric across documents.

    Args:
        df: DataFrame containing metrics (should have 'method' and metric columns)
        metric: Metric to plot ('wer', 'cer', 'bwer', etc.)
        by_method: Whether to group by method
        output_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 8))

    # Set seaborn style
    sns.set(style="whitegrid")

    # Create distribution plot
    if by_method and 'method' in df.columns:
        ax = sns.boxplot(x='method', y=metric, data=df)
        sns.stripplot(x='method', y=metric, data=df,
                    size=4, color=".3", alpha=0.7)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Set labels
        plt.ylabel(metric.upper())
        plt.title(f'Distribution of {metric.upper()} by Method')
    else:
        # Simple histogram if not grouping by method
        sns.histplot(df[metric], kde=True)
        plt.xlabel(metric.upper())
        plt.title(f'Distribution of {metric.upper()}')

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def plot_line_metrics(document_id: str, line_results_path: str,
                     metrics: List[str] = ['cer', 'wer', 'bwer'],
                     output_path: Optional[str] = None):
    """
    Plot line-level metrics for a single document.

    Args:
        document_id: Document identifier
        line_results_path: Path to the line results CSV file
        metrics: List of metrics to plot
        output_path: Optional path to save the figure
    """
    # Load line results
    df = pd.read_csv(line_results_path)

    # Set up plot
    plt.figure(figsize=(14, 6))

    # Create a subplot for each metric
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)

        # Plot line metrics
        sns.barplot(x='line_number', y=metric, data=df)

        # Set labels
        plt.title(f'Line-level {metric.upper()}')
        plt.xlabel('Line Number')
        plt.ylabel(metric.upper())

    plt.suptitle(f'Line-level Metrics for Document {document_id}', fontsize=16)
    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def create_confusion_matrix(line_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Create a confusion matrix of predicted vs actual text for error analysis.
    This works at the character or word level to show patterns of errors.

    Args:
        line_df: DataFrame with line results (must have 'ground_truth' and 'transcription' columns)
        output_path: Optional path to save the figure
    """
    from collections import Counter

    # Combine all ground truth and transcription
    all_gt = ' '.join(line_df['ground_truth'].tolist())
    all_tr = ' '.join(line_df['transcription'].tolist())

    # Get unique characters or words
    # For character-level analysis:
    gt_chars = set(all_gt)
    tr_chars = set(all_tr)
    unique_chars = sorted(list(gt_chars.union(tr_chars)))

    # Count character substitutions
    substitutions = Counter()

    for gt_line, tr_line in zip(line_df['ground_truth'], line_df['transcription']):
        # Compare the minimum length to avoid index errors
        min_len = min(len(gt_line), len(tr_line))

        for i in range(min_len):
            if gt_line[i] != tr_line[i]:
                substitutions[(gt_line[i], tr_line[i])] += 1

    # Get top substitutions
    top_subs = substitutions.most_common(15)

    # Plot confusion matrix for top substitutions
    plt.figure(figsize=(10, 8))

    if top_subs:
        chars_gt, chars_tr = zip(*[t[0] for t in top_subs])
        counts = [t[1] for t in top_subs]

        # Create DataFrame for plotting
        sub_df = pd.DataFrame({
            'GT': chars_gt,
            'Prediction': chars_tr,
            'Count': counts
        })

        # Create a pivot table
        pivot_df = sub_df.pivot_table(index='GT', columns='Prediction', values='Count', fill_value=0)

        # Plot heatmap
        sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Character Substitution Patterns')
    else:
        plt.text(0.5, 0.5, "No substitution errors found",
                horizontalalignment='center', fontsize=14)

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def plot_method_comparison(results_dir: str, methods: List[str],
                          metrics: List[str] = ['wer', 'cer', 'bwer', 'wer_diff'],
                          output_path: Optional[str] = None):
    """
    Plot a comparison of metrics across different methods.

    Args:
        results_dir: Directory containing results
        methods: List of method names
        metrics: List of metrics to plot
        output_path: Optional path to save the figure
    """
    # Load aggregate results for each method
    aggregates = []

    for method in methods:
        agg_path = os.path.join(results_dir, f"{method}_aggregate_results.csv")
        if os.path.exists(agg_path):
            agg_df = pd.read_csv(agg_path)
            if not agg_df.empty:
                aggregates.append(agg_df)

    if not aggregates:
        print("No aggregate results found")
        return None

    # Combine all aggregates
    combined = pd.concat(aggregates)

    # Set up plot
    plt.figure(figsize=(14, 8))

    # Plot each metric
    for i, metric in enumerate(metrics):
        # Skip if metric not present
        if metric not in combined.columns and f'mean_{metric}' not in combined.columns:
            continue

        # Get metric column name
        col_name = metric if metric in combined.columns else f'mean_{metric}'

        # Get error bars if available
        error_col = f'std_{metric}' if f'std_{metric}' in combined.columns else None

        plt.subplot(1, len(metrics), i+1)

        # Create bar plot with error bars if available
        if error_col:
            ax = sns.barplot(x='method', y=col_name, data=combined,
                           yerr=combined[error_col])
        else:
            ax = sns.barplot(x='method', y=col_name, data=combined)

        # Add value labels on top of bars
        for j, v in enumerate(combined[col_name]):
            ax.text(j, v + 0.01, f"{v:.3f}", ha='center')

        # Set labels
        plt.title(f'{metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xlabel('Method')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

    plt.suptitle('Comparison of Methods', fontsize=16)
    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def plot_line_length_vs_error(line_results_dir: str, methods: List[str],
                             metric: str = 'wer', output_path: Optional[str] = None):
    """
    Plot the relationship between line length and error rates.

    Args:
        line_results_dir: Directory containing line-level results
        methods: List of methods to include
        metric: Metric to plot against line length
        output_path: Optional path to save the figure
    """
    # Load all line results
    all_lines = []

    for method in methods:
        method_dir = os.path.join(line_results_dir, method)
        if not os.path.exists(method_dir):
            continue

        for filename in os.listdir(method_dir):
            if filename.endswith('_line_results.csv'):
                file_path = os.path.join(method_dir, filename)
                df = pd.read_csv(file_path)

                # Add line length
                df['line_length'] = df['ground_truth'].apply(len)

                # Add method if not present
                if 'method' not in df.columns:
                    df['method'] = method

                all_lines.append(df)

    if not all_lines:
        print("No line results found")
        return None

    # Combine all line results
    combined = pd.concat(all_lines)

    # Create scatter plot with regression line
    plt.figure(figsize=(12, 8))

    # Use different colors for different methods
    sns.lmplot(x='line_length', y=metric, hue='method', data=combined,
              scatter_kws={'alpha': 0.5}, height=6, aspect=1.5)

    # Set labels
    plt.title(f'Line Length vs {metric.upper()}')
    plt.xlabel('Line Length (characters)')
    plt.ylabel(metric.upper())

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def create_error_heatmap(results_df: pd.DataFrame, doc_column: str = 'document_id',
                        method_column: str = 'method', error_column: str = 'wer',
                        output_path: Optional[str] = None):
    """
    Create a heatmap showing error rates by document and method.

    Args:
        results_df: DataFrame with results
        doc_column: Column name for document identifiers
        method_column: Column name for method identifiers
        error_column: Column name for the error metric
        output_path: Optional path to save the figure
    """
    # Create pivot table
    pivot_df = results_df.pivot(index=doc_column, columns=method_column, values=error_column)

    # Sort by average error rate (better visualization)
    pivot_df['avg'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg')
    pivot_df = pivot_df.drop('avg', axis=1)

    # Set up plot
    plt.figure(figsize=(12, max(8, len(pivot_df) * 0.3)))

    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu_r",
               linewidths=0.5, cbar_kws={'label': error_column.upper()})

    # Set labels
    plt.title(f'{error_column.upper()} by Document and Method')
    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def visualize_example_document(document_id: str, ground_truth_lines: List[str],
                              method_results: Dict[str, List[str]],
                              output_path: Optional[str] = None):
    """
    Create a visual comparison of transcriptions from different methods.

    Args:
        document_id: Document identifier
        ground_truth_lines: List of ground truth text lines
        method_results: Dict mapping method names to lists of transcribed lines
        output_path: Optional path to save the figure
    """
    import difflib
    from matplotlib.colors import ListedColormap

    # Set up the plot
    n_methods = len(method_results)
    plt.figure(figsize=(15, 4 + n_methods * 3))

    # Show ground truth at the top
    plt.subplot(n_methods + 1, 1, 1)
    plt.text(0.5, 0.5, '\n'.join(ground_truth_lines),
            horizontalalignment='center', verticalalignment='center',
            fontfamily='monospace', wrap=True)
    plt.title('Ground Truth')
    plt.axis('off')

    # For each method, show its transcription with differences highlighted
    for i, (method_name, method_lines) in enumerate(method_results.items(), 1):
        plt.subplot(n_methods + 1, 1, i + 1)

        # Join lines
        gt_text = '\n'.join(ground_truth_lines)
        method_text = '\n'.join(method_lines[:len(ground_truth_lines)])

        # Create comparison text with HTML-like markup
        d = difflib.Differ()
        diff = list(d.compare(gt_text.split(), method_text.split()))

        # Format with highlighting
        formatted = []
        for word in diff:
            if word.startswith('+ '):
                formatted.append(f'<span style="color:green">{word[2:]}</span>')
            elif word.startswith('- '):
                formatted.append(f'<span style="color:red">{word[2:]}</span>')
            elif word.startswith('  '):
                formatted.append(word[2:])

        # Display with simple formatting (matplotlib has limited HTML support)
        plt.text(0.5, 0.5, ' '.join(formatted),
                horizontalalignment='center', verticalalignment='center',
                fontfamily='monospace', wrap=True)
        plt.title(method_name)
        plt.axis('off')

    plt.suptitle(f'Transcription Comparison for Document {document_id}', fontsize=16)
    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def plot_reading_order_error(results_df: pd.DataFrame, by_method: bool = True,
                           output_path: Optional[str] = None):
    """
    Plot the WER-bWER difference which correlates with reading order errors.

    Args:
        results_df: DataFrame with results (must have 'wer' and 'bwer' columns)
        by_method: Whether to group by method
        output_path: Optional path to save figure
    """
    # Calculate WER-bWER difference
    results_df['wer_diff'] = results_df['wer'] - results_df['bwer']

    # Create plot similar to plot_metric_distribution but for this specific metric
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    if by_method and 'method' in results_df.columns:
        ax = sns.boxplot(x='method', y='wer_diff', data=results_df)
        sns.stripplot(x='method', y='wer_diff', data=results_df,
                     size=4, color=".3", alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.title('Reading Order Error (WER-bWER) by Method')
        plt.ylabel('WER-bWER')
    else:
        sns.histplot(results_df['wer_diff'], kde=True)
        plt.xlabel('WER-bWER')
        plt.title('Distribution of Reading Order Errors')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def plot_error_type_breakdown(results_df: pd.DataFrame, methods: List[str] = None,
                             output_path: Optional[str] = None):
    """
    Visualize the breakdown of errors into content recognition errors (bWER)
    and reading order errors (WER-bWER).

    Args:
        results_df: DataFrame with results (must have 'wer' and 'bwer' columns)
        methods: List of methods to include (None for all)
        output_path: Optional path to save figure
    """
    # Filter by methods if specified
    if methods and 'method' in results_df.columns:
        df = results_df[results_df['method'].isin(methods)].copy()
    else:
        df = results_df.copy()

    # Calculate the two error components
    df['content_errors'] = df['bwer']
    df['order_errors'] = df['wer'] - df['bwer']

    # Reshape for stacked bar chart
    plot_df = pd.melt(df,
                     id_vars=['method'] if 'method' in df.columns else [],
                     value_vars=['content_errors', 'order_errors'],
                     var_name='Error Type', value_name='Error Rate')

    # Create stacked bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='method' if 'method' in df.columns else None,
               y='Error Rate', hue='Error Type', data=plot_df)

    plt.title('Breakdown of Content vs. Reading Order Errors')
    if 'method' in df.columns:
        plt.xlabel('Method')
        plt.xticks(rotation=45, ha='right')

    plt.ylabel('Error Rate')
    plt.legend(title='Error Type')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def plot_metric_correlation(results_df: pd.DataFrame, x_metric: str = 'wer',
                           y_metric: str = 'bwer', output_path: Optional[str] = None):
    """
    Create a scatter plot showing correlation between two metrics.

    Args:
        results_df: DataFrame with results
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        output_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))

    # Create scatter plot with regression line
    sns.regplot(x=x_metric, y=y_metric, data=results_df,
               scatter_kws={'alpha': 0.7})

    # Add diagonal line (y=x) for reference
    min_val = min(results_df[x_metric].min(), results_df[y_metric].min())
    max_val = max(results_df[x_metric].max(), results_df[y_metric].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

    # Calculate and display correlation coefficient
    corr = results_df[[x_metric, y_metric]].corr().iloc[0, 1]
    plt.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')

    plt.title(f'Correlation Between {x_metric.upper()} and {y_metric.upper()}')
    plt.xlabel(x_metric.upper())
    plt.ylabel(y_metric.upper())
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return plt.gcf()
