# # evaluate.py
# import os
# import argparse
# import trackeval

# def main():
#     parser = argparse.ArgumentParser(description='Evaluate tracking results using TrackEval.')
#     parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth directory.')
#     parser.add_argument('--results_path', type=str, required=True, help='Path to the tracker results directory.')
#     parser.add_argument('--out_dir', type=str, default='eval_output', help='Directory to save evaluation results.')
#     args = parser.parse_args()
    
#     # --- Configuration for TrackEval ---
#     eval_config = trackeval.Evaluator.get_default_eval_config()
#     # You can disable default metrics if you only want HOTA, etc.
#     # eval_config['METRICS'] = {'HOTA', 'CLEAR', 'Identity'}
    
#     dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
#     dataset_config['GT_FOLDER'] = args.gt_path
#     dataset_config['TRACKERS_FOLDER'] = args.results_path
#     dataset_config['TRACKERS_TO_EVAL'] = ['']  # Evaluate the tracker in the root of TRACKERS_FOLDER
#     dataset_config['SPLIT_TO_EVAL'] = 'train' # The default split for MOTChallenge
#     dataset_config['BENCHMARK'] = 'MOT17' # Use a standard benchmark for metric names
    
#     # By default, TrackEval expects gt.txt inside a sequence folder, e.g., GT_FOLDER/SEQ_NAME/gt/gt.txt
#     # We will tell it our gt.txt is directly in the GT_FOLDER
#     dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/gt.txt'

#     # --- Run Evaluation ---
#     evaluator = trackeval.Evaluator(eval_config)
#     dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
#     print("Running evaluation...")
#     output_res, output_msg = evaluator.evaluate(dataset_list,-1)
    
#     # The metrics are in a nested dictionary, let's print them
#     metrics = list(output_res['MotChallenge2DBox']['']['']['COMBINED_SEQ'].keys())
    
#     print("\n--- Evaluation Summary ---")
#     for metric in metrics:
#         val = output_res['MotChallenge2DBox']['']['']['COMBINED_SEQ'][metric]
#         print(f"{metric:<15}: {val:.3f}")
        
#     print("\n✅ Evaluation complete.")

# if __name__ == '__main__':
#     main()

# tools/evaluate.py (with visualization)
# tools/evaluate.py (Corrected for custom datasets)
import os
import argparse
import trackeval
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize tracking results.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the directory containing the ground truth file.')
    parser.add_argument('--results_path', type=str, required=True, help='Path to the directory containing the tracker results file.')
    parser.add_argument('--out_dir', type=str, default='evaluation/final_report', help='Directory to save evaluation results and plot.')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # --- Configuration for TrackEval ---
    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    
    dataset_config['GT_FOLDER'] = args.gt_path
    dataset_config['TRACKERS_FOLDER'] = args.results_path
    dataset_config['TRACKERS_TO_EVAL'] = [''] 
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    
    # --- KEY CHANGES FOR CUSTOM DATASET ---
    # 1. Set BENCHMARK to 'Custom' to stop it from looking for benchmark-specific files.
    dataset_config['BENCHMARK'] = 'Custom'
    
    # 2. Provide a name for your custom sequence. The library needs at least one sequence name.
    #    The 'None' value means the sequence length will be inferred from the data.
    dataset_config['SEQ_INFO'] = {'drone-sequence': None}
    
    # 3. Tell the evaluator where to find the ground truth file directly.
    dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/gt.txt'

    # --- Run Evaluation ---
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    print("Running evaluation...")
    output_res, output_msg = evaluator.evaluate(dataset_list)
    
    # --- 1. Print the Detailed Text Summary ---
    # The results are now nested under the custom sequence name 'drone-sequence'
    results_dict = output_res['MotChallenge2DBox']['']['drone-sequence']
    
    mh = trackeval.metrics.create_metrics(eval_config['METRICS'])
    summary = mh.compute_unique_metrics(results_dict)
    strsummary = trackeval.io.render_summary(summary, formatters=mh.get_display_name_formatters())
    print("\n--- Detailed Evaluation Summary ---")
    print(strsummary)
    
    # --- 2. Visualize the Key Metrics ---
    print("\nGenerating metrics plot...")
    
    metrics_to_plot = {
        'HOTA': summary['HOTA']['HOTA'],
        'MOTA': summary['CLEAR']['MOTA'],
        'IDF1': summary['Identity']['IDF1'],
        'AssA': summary['HOTA']['AssA'],
        'DetA': summary['HOTA']['DetA'],
        'MOTP': summary['CLEAR']['MOTP'],
    }
    
    names = list(metrics_to_plot.keys())
    values = [v * 100 for v in metrics_to_plot.values()]

    plt.figure(figsize=(12, 7))
    sns.barplot(x=values, y=names, palette='viridis', orient='h')
    
    for index, value in enumerate(values):
        plt.text(value, index, f' {value:.2f}%', va='center', fontsize=10)
        
    plt.title('Tracking Performance Metrics Summary', fontsize=16)
    plt.xlabel('Score (%)', fontsize=12)
    plt.xlim(0, 105)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(args.out_dir, 'evaluation_summary.png')
    plt.savefig(plot_path, bbox_inches='tight')
    
    print(f"✅ Evaluation complete. Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
