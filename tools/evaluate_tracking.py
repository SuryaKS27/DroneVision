# evaluate.py
import os
import argparse
import trackeval

def main():
    parser = argparse.ArgumentParser(description='Evaluate tracking results using TrackEval.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth directory.')
    parser.add_argument('--results_path', type=str, required=True, help='Path to the tracker results directory.')
    parser.add_argument('--out_dir', type=str, default='eval_output', help='Directory to save evaluation results.')
    args = parser.parse_args()
    
    # --- Configuration for TrackEval ---
    eval_config = trackeval.Evaluator.get_default_eval_config()
    # You can disable default metrics if you only want HOTA, etc.
    # eval_config['METRICS'] = {'HOTA', 'CLEAR', 'Identity'}
    
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = args.gt_path
    dataset_config['TRACKERS_FOLDER'] = args.results_path
    dataset_config['TRACKERS_TO_EVAL'] = ['']  # Evaluate the tracker in the root of TRACKERS_FOLDER
    dataset_config['SPLIT_TO_EVAL'] = 'train' # The default split for MOTChallenge
    dataset_config['BENCHMARK'] = 'MOT17' # Use a standard benchmark for metric names
    
    # By default, TrackEval expects gt.txt inside a sequence folder, e.g., GT_FOLDER/SEQ_NAME/gt/gt.txt
    # We will tell it our gt.txt is directly in the GT_FOLDER
    dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/gt.txt'

    # --- Run Evaluation ---
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    print("Running evaluation...")
    output_res, output_msg = evaluator.evaluate(dataset_list,-1)
    
    # The metrics are in a nested dictionary, let's print them
    metrics = list(output_res['MotChallenge2DBox']['']['']['COMBINED_SEQ'].keys())
    
    print("\n--- Evaluation Summary ---")
    for metric in metrics:
        val = output_res['MotChallenge2DBox']['']['']['COMBINED_SEQ'][metric]
        print(f"{metric:<15}: {val:.3f}")
        
    print("\n✅ Evaluation complete.")

if __name__ == '__main__':
    main()
