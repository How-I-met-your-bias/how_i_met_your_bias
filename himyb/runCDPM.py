import argparse
import os
from DiffusionFreeGuidance.TrainCondition import train, eval

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "state": "train",  # or eval
    "iterations": 100000,
    "batch_size": 50,
    "T": 1000,
    "channel": 128,
    "channel_mult": [1, 2, 2, 2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.5,
    "beta_1": 1e-4,
    "beta_T": 0.028,
    "img_size": 64, # 64 for bffhq and 32 for bmnist
    "grad_clip": 1.0,
    "device": "cuda:0",
    # "save_dir": "./ModelCheckpoints/conditioned", # Directory to save model checkpoints
    "save_dir": "/data05/ani/final_checkpoints_cdpm", # Directory to save model checkpoints

    "w": 1,
    "load_weights": None,
    "sampled_dir": "/data05/ani/GenImages/conditioned/DpmSolver++150s_3o", # Directory to save sampled images
    "images_to_sample": 1000,
    "nrow": 1,
    "freq_save": 10000,  # Save every 10k iterations
    "dataset": "bffhq", # bmnist
    "data_dir": "./data",
    "continue_from_existing": False,
    "dpm-solver++": True,
    "dpm_steps": 150,    # Fast sampling with 150 steps
    "dpm_order": 3,     # Use 3rd order for better quality
}

def parse_args():
    """Parse command-line arguments to override default configurations."""
    parser = argparse.ArgumentParser(description="Model Configuration")
    
    # Add arguments corresponding to the model config keys
    parser.add_argument("--state", type=str, choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--T", type=int, help="Number of time steps")
    parser.add_argument("--channel", type=int, help="Base number of channels")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--img_size", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device to use, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--w", type=float, help="Classifier-free guided diffusion strength")
    parser.add_argument("--save_dir", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--freq_save", type=int, help="Iteration frequency to save model checkpoints")
    parser.add_argument("--load_weights", type=str, help="Path to model checkpoint to be used for evaluation")
    parser.add_argument("--sampled_dir", type=str, help="Directory to save sampled images")
    parser.add_argument("--images_to_sample", type=int, help="Number of images to sample")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--data_dir", type=str, help="Directory containing dataset")
    
    # Add new arguments for parameter sweep
    parser.add_argument("--enable_sweep", action="store_true", help="Enable parameter sweep mode")
    parser.add_argument("--T_values", nargs='+', type=int, default=[600, 700, 800, 900, 1000], # [600, 700, 800, 900, 1000]
                        help="List of T values to train with")
    parser.add_argument("--w_values", nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0], # [0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]
                        help="List of w values to sample with")

    return parser.parse_args()

def update_config(default_config, args):
    """Update the default configuration with command-line arguments."""
    config = default_config.copy()
    for key, value in vars(args).items():
        if value is not None and key not in ['enable_sweep', 'T_values', 'w_values']:
            config[key] = value
    return config

def train_model(config, T_val):
    """Train a model with given T value."""
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL WITH T = {T_val}")
    print(f"{'='*60}")
    
    # Update config for training
    train_config = config.copy()
    train_config["T"] = T_val
    train_config["state"] = "train"
    
    # Set unique save directory for this T value
    train_config["save_dir"] = f"{config['save_dir']}/T_{T_val}"
    
    print(f"Training Configuration:")
    print(f"  T: {train_config['T']}")
    print(f"  Iterations: {train_config['iterations']}")
    print(f"  Batch Size: {train_config['batch_size']}")
    print(f"  Save Directory: {train_config['save_dir']}")
    
    # Create save directory if it doesn't exist
    os.makedirs(train_config["save_dir"], exist_ok=True)
    
    try:
        print("Starting training...")
        train(train_config)
        print(f"Training completed for T = {T_val}")
        return True
    except Exception as e:
        print(f"Error during training with T = {T_val}: {e}")
        return False

def sample_with_w(config, T_val, w_val, checkpoint_path):
    """Sample images with given w value using trained checkpoint."""
    print(f"\n    Sampling with w = {w_val}...")
    
    # Update config for sampling
    sample_config = config.copy()
    sample_config["T"] = T_val
    sample_config["w"] = w_val
    sample_config["state"] = "eval"
    sample_config["load_weights"] = checkpoint_path
    
    # Set unique sample directory for this T and w combination
    # Replace dot with underscore in w_val for folder name
    sample_config["sampled_dir"] = f"{config['sampled_dir']}/{config['dataset']}/T_{T_val}/w_{w_val}/"
    
    # Create sample directory if it doesn't exist
    os.makedirs(sample_config["sampled_dir"], exist_ok=True)
    
    print(f"      Sample Directory: {sample_config['sampled_dir']}")
    print(f"      Images to Sample: {sample_config['images_to_sample']}")
    
    try:
        eval(sample_config)
        print(f"      Sampling completed for w = {w_val}")
        return True
    except Exception as e:
        print(f"      Error during sampling with w = {w_val}: {e}")
        return False

def run_complete_experiment(config, T_values, w_values):
    """Run complete experiment: train with each T, then sample with each w."""
    total_T_values = len(T_values)
    total_w_values = len(w_values)
    
    print(f"\n{'='*80}")
    print(f"STARTING COMPLETE EXPERIMENT")
    print(f"Training with T values: {T_values}")
    print(f"Sampling with w values: {w_values}")
    print(f"Total models to train: {total_T_values}")
    print(f"Total sampling runs: {total_T_values * total_w_values}")
    print(f"{'='*80}")
    
    successful_trains = 0
    successful_samples = 0
    
    # Outer loop: Train with each T value
    for i, T_val in enumerate(T_values):
        print(f"\n[{i+1}/{total_T_values}] Processing T = {T_val}")
        
        # Train the model
        # training_success = train_model(config, T_val)
        training_success = True
        
        if training_success:
            successful_trains += 1
            
            if config['dataset'] == 'bmnist':
                # Define checkpoint path (adjust based on your checkpoint naming convention)
                checkpoint_path = f"{config['save_dir']}/T_{T_val}/{config['dataset']}/ckpt_{config['iterations']}_iterations.pt"

            else:
                checkpoint_path = f"{config['save_dir']}/T_{T_val}/ckpt_{config['iterations']}_iterations.pt"
                
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                # Try alternative checkpoint names
                alt_paths = [
                    f"{config['save_dir']}/T_{T_val}/{config['dataset']}/ckpt_{config['freq_save']}_iterations.pt",
                ]
                
                checkpoint_found = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        checkpoint_path = alt_path
                        checkpoint_found = True
                        break
                
                if not checkpoint_found:
                    print(f"    Warning: No checkpoint found for T = {T_val}")
                    print(f"    Checked paths: {checkpoint_path}")
                    for alt_path in alt_paths:
                        print(f"                   {alt_path}")
                    continue
            
            print(f"    Using checkpoint: {checkpoint_path}")
            print(f"    Starting sampling with {len(w_values)} different w values...")
            
            # Inner loop: Sample with each w value
            for j, w_val in enumerate(w_values):
                print(f"    [{j+1}/{total_w_values}] ", end="")
                
                sampling_success = sample_with_w(config, T_val, w_val, checkpoint_path)
                
                if sampling_success:
                    successful_samples += 1
            
            print(f"    Completed sampling for T = {T_val}")
        else:
            print(f"    Skipping sampling for T = {T_val} due to training failure")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED!")
    print(f"Successful trainings: {successful_trains}/{total_T_values}")
    print(f"Successful samplings: {successful_samples}/{total_T_values * total_w_values}")
    print(f"{'='*80}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Update configuration
    modelConfig = update_config(DEFAULT_MODEL_CONFIG, args)
    
    if args.enable_sweep:
        print("Parameter sweep mode enabled!")
        print(f"T values for training: {args.T_values}")
        print(f"w values for sampling: {args.w_values}")
        
        # Run complete experiment
        run_complete_experiment(modelConfig, args.T_values, args.w_values)
        
    else:
        # Original single run mode
        print(f"Single run mode - Experiment Configuration: {modelConfig}")
        
        # Run train or eval based on the state
        if modelConfig["state"] == "train":
            train(modelConfig)
        else:
            eval(modelConfig)

if __name__ == '__main__':
    main()