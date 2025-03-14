# Configuration and profiling
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
from wandb.apis import public

import pandas as pd
import os
import json
import yaml
import cProfile

@hydra.main(config_path="../configs", config_name="config_default.yaml")
def main(config: DictConfig):
    # Resolve Hydra config
    config = OmegaConf.to_container(config, resolve=True)

    # W&B Project Information
    wandb_project = config["project"]
    entity = config.get("entity", None)  # Optional: If using an organization

    # Export directory
    export_dir = config.get("export_dir", "data/wandb_exports")
    os.makedirs(export_dir, exist_ok=True)

    print(f"Fetching runs from W&B project: {wandb_project}")

    # Initialize W&B API
    api = wandb.Api()

    # Fetch runs
    runs = api.runs(f"{entity}/{wandb_project}" if entity else wandb_project)

    for run in runs:
        try:
            run : public.Run
            run_id = run.id
            run_name = run.name or run_id  # Use run name if available
            run_path = os.path.join(export_dir, run_name)

            os.makedirs(run_path, exist_ok=True)

            print(f"Exporting run: {run_name} (ID: {run_id})")

            # Export Metrics
            df = pd.DataFrame(run.history(samples=10000))  # Adjust sample size if needed
            metrics_path = os.path.join(run_path, "metrics.csv")
            df.to_csv(metrics_path, index=False)
            print(f"Saved metrics to {metrics_path}")

            # Export Config
            config_path_json = os.path.join(run_path, "config.json")
            with open(config_path_json, "w") as f:
                json.dump(run.config, f, indent=4)
            print(f"Saved config to {config_path_json}")

            # Optional: Save config as YAML
            config_path_yaml = os.path.join(run_path, "config.yaml")
            with open(config_path_yaml, "w") as f:
                yaml.dump(run.config, f, default_flow_style=False)
            print(f"Saved config to {config_path_yaml}")
        except:
            breakpoint()
        breakpoint()
        
    print("Export complete.")

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    log_file_cprofile = "logs/profile_stats.prof"
    pr.dump_stats(log_file_cprofile)
    print(f"[PROFILING] Profile stats dumped to {log_file_cprofile}. You can visualize it using 'snakeviz {log_file_cprofile}'")
