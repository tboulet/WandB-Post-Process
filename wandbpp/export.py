import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
from wandb.apis import public

import pandas as pd
import numpy as np
import os
import yaml
import json
import cProfile

def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@hydra.main(config_path="../configs", config_name="config_default.yaml")
def main(config: DictConfig):
    # Resolve Hydra config
    config = OmegaConf.to_container(config, resolve=True)

    # W&B Project Information
    wandb_project = config["project"]
    entity = config.get("entity", None)
    filters = config.get("filters", None)
    
    # Export directory
    export_dir = config.get("export_dir", "data/wandb_exports")
    os.makedirs(export_dir, exist_ok=True)

    print(f"Fetching runs from W&B project: {wandb_project}")
    
    api = wandb.Api()
    runs = api.runs(
        path=f"{entity}/{wandb_project}" if entity else wandb_project,
        filters=filters,
    )

    for run in runs:
        try:
            run: public.Run
            run_id = run.id
            run_name = run.name or run_id
            run_path = os.path.join(export_dir, run_name)

            df = run.history(samples=10000, pandas=True)

            # Apply filters
            if "_step" in df.columns and df["_step"].max() < 1000:
                print(f"Skipping run {run_name} (max _step: {df['_step'].max()})")
                continue

            if len(df.columns) < 10:
                print(f"Skipping run {run_name} (only {len(df.columns)} metrics logged)")
                continue
            
            os.makedirs(run_path, exist_ok=True)
            print(f"Exporting run: {run_name} (ID: {run_id})")

            # Separate scalars and histograms
            scalars = {}
            histograms = {}
            images = {}
            
            for col in df.columns:
                sample_value = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
                
                if isinstance(sample_value, dict) and "_type" in sample_value:
                    if sample_value["_type"] == "histogram":
                        histograms[col] = sample_value
                    elif sample_value["_type"] == "image-file":
                        images[col] = sample_value
                else:
                    scalars[col] = df[col].dropna().tolist()

            # Save scalars (metrics) to CSV
            scalar_df = df[list(scalars.keys())]
            metrics_path = os.path.join(run_path, "metrics.csv")
            scalar_df.to_csv(metrics_path, index=False)
            print(f"Saved metrics to {metrics_path}")

            # Save histograms to JSON
            histograms_path = os.path.join(run_path, "histograms.json")
            with open(histograms_path, "w") as f:
                json.dump(histograms, f, indent=4, default=convert_numpy)
            print(f"Saved histograms to {histograms_path}")

            # Save images to JSON
            images_path = os.path.join(run_path, "images.json")
            with open(images_path, "w") as f:
                json.dump(images, f, indent=4, default=convert_numpy)
            print(f"Saved images to {images_path}")

            # Save config as YAML
            config_path_yaml = os.path.join(run_path, "config.yaml")
            with open(config_path_yaml, "w") as f:
                yaml.dump(run.config, f, default_flow_style=False)
            print(f"Saved config to {config_path_yaml}")
            breakpoint()
        except Exception as e:
            print(f"Error processing run {run_name}: {e}")
    
    print("Export complete.")

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    log_file_cprofile = "logs/profile_stats.prof"
    pr.dump_stats(log_file_cprofile)
    print(f"[PROFILING] Profile stats dumped to {log_file_cprofile}. You can visualize it using 'snakeviz {log_file_cprofile}'")
