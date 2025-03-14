# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
import cProfile



@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)
    
    project = config["project"]
    print(f"Project: {project}")
    

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    log_file_cprofile = "logs/profile_stats.prof"
    pr.dump_stats(log_file_cprofile)
    print(f"[PROFILING] Profile stats dumped to {log_file_cprofile}. You can visualize the profile stats using snakeviz by running 'snakeviz {log_file_cprofile}'")
