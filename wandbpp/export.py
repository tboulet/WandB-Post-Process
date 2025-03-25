from collections import defaultdict
import logging
import sys
from typing import Dict
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

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@hydra.main(config_path="../configs_export", config_name="config_default.yaml")
def main(config: DictConfig):
    # Resolve Hydra config
    config = OmegaConf.to_container(config, resolve=True)

    # Config options
    wandb_project = config["project"]
    entity = config.get("entity", "")
    filters = config.get("filters", None)
    samples = config.get("samples", 10000)
    min_n_metrics = config.get("min_n_metrics", 1)
    data_types : Dict[str, bool] = config["data_types"]
    
    # Export directory
    export_dir = config.get("export_dir", "data/wandbpp_exports")
    os.makedirs(export_dir, exist_ok=True)

    logger.info(f"Fetching runs from W&B project: {wandb_project}...")    
    
    api = wandb.Api()
    runs = api.runs(
        path=os.path.join(entity, wandb_project),
        filters=filters,
    )

    logger.info(f"Found {len(runs)} runs in project {wandb_project}.")
    logger.info(f"Exporting to {export_dir}...")
    
    for run in runs:
        try:
            run: public.Run
            run_id = run.id
            run_name = run.name or run_id
            safe_run_name = run_name.replace("/", "_")
            run_path = os.path.join(export_dir, safe_run_name)            
            run_url = os.path.join("https://api.wandb.ai/files", entity, wandb_project, run_id)
            
            df = run.history(samples=samples, pandas=True)

            if len(df.columns) < min_n_metrics:
                logger.info(f"Skipping run {run_name} (only {len(df.columns)} metrics logged)")
                continue
            
            os.makedirs(run_path, exist_ok=True)
            logger.info(f"Exporting run: {run_name} (ID: {run_id})")

            # Get the last logged step as reference
            step_max = df["_step"].max() if "_step" in df.columns else None
            last_logged_data = df[df["_step"] == step_max].iloc[-1] if step_max is not None else None
            
            # Save metadata
            if data_types["metadata"]:
                metadata_path = os.path.join(run_path, "metadata.yaml")
                with open(metadata_path, "w") as f:
                    yaml.dump({
                        "id": run_id,
                        "name": run_name,
                        "url": run_url,
                        "step_max": convert_numpy(step_max),
                    }, f, default_flow_style=False)
                logger.info(f"Saved metadata to {metadata_path}")
                
            # Separate scalars, histograms, and images
            scalars = {}
            histograms = {}
            images_urls = {}
            
            for col in df.columns:
                if last_logged_data is not None and col in last_logged_data:
                    sample_value = last_logged_data[col]
                    if isinstance(sample_value, dict) and "_type" in sample_value:
                        if data_types["histogram"] and sample_value["_type"] == "histogram":
                            histograms[col] = df[col].dropna().tolist()
                        elif data_types["image_url"] and sample_value["_type"] == "image-file":
                            images_urls[col] = df[col].dropna().tolist()
                    elif data_types["scalar"]:
                        scalars[col] = df[col].dropna().tolist()
                
            # Save scalars to CSV
            if data_types["scalar"]:
                scalar_df = df[list(scalars.keys())]
                scalars_path = os.path.join(run_path, "scalars.csv")
                scalar_df.to_csv(scalars_path, index=False)
                logger.info(f"Saved scalars to {scalars_path}")

            # Save histograms to JSON
            if data_types["histogram"]:
                histograms_path = os.path.join(run_path, "histograms.json")
                with open(histograms_path, "w") as f:
                    json.dump(histograms, f, indent=4, default=convert_numpy)
                logger.info(f"Saved histograms to {histograms_path}")

            # Save images URLs to JSON
            if data_types["image_url"]:
                images_urls_path = os.path.join(run_path, "images_urls.json")
                with open(images_urls_path, "w") as f:
                    json.dump(images_urls, f, indent=4, default=convert_numpy)
                logger.info(f"Saved images URLs to {images_urls_path}")

            # Save config as YAML
            if data_types["config"]:
                config_path_yaml = os.path.join(run_path, "config.yaml")
                with open(config_path_yaml, "w") as f:
                    yaml.dump(run.config, f, default_flow_style=False)
                logger.info(f"Saved config to {config_path_yaml}")
            
            # breakpoint()
            
        except Exception as e:
            logger.error(f"Error processing run {run_name}: {e}")
    
    logger.info("Export complete.")

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    log_file_cprofile = "logs/profile_stats.prof"
    os.makedirs("logs", exist_ok=True)
    pr.dump_stats(log_file_cprofile)
    logger.info(f"[PROFILING] Profile stats dumped to {log_file_cprofile}. You can visualize it using 'snakeviz {log_file_cprofile}'")
