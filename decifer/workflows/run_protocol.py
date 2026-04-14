from decifer.experimental import DeciferPipeline
from decifer.config import RunProtocolConfig, load_dataclass_config
from decifer.io import create_run_layout
import argparse
import os

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-path", dest="model_path", type=str, default=None)
    parser.add_argument("--zip-path", dest="zip_path", type=str, default=None)
    parser.add_argument("--debug-max", dest="debug_max", type=int, default=None)
    parser.add_argument("--n-trials", dest="n_trials", type=int, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    return parser


def parse_config(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    config = load_dataclass_config(RunProtocolConfig, config_path=config_path, overrides=overrides)

    if not config.model_path:
        raise ValueError("The 'model_path' option is required and cannot be empty")
    if not config.zip_path:
        raise ValueError("The 'zip_path' option is required and cannot be empty")

    return config


def resolve_run_dir(config: RunProtocolConfig) -> str:
    return os.path.abspath(f"particles_CeO2_protocol_{config.suffix}")


def main(argv=None):
    config = parse_config(argv)
    run_dir = resolve_run_dir(config)
    layout = create_run_layout(
        run_dir,
        "run_protocol",
        config,
        metadata={
            "model_path": os.path.abspath(config.model_path),
            "zip_path": os.path.abspath(config.zip_path),
        },
    )

    # Create pipeline
    pipeline = DeciferPipeline(
        model_path = config.model_path,
        zip_path = config.zip_path,
        results_output_folder=layout.predictions_dir,
    )

    pipeline.setup_folder(layout.predictions_dir)

    target_files = config.target_files
    if config.debug_max is not None:
        target_files = target_files[:config.debug_max]

    generated_pickles = []
    for target_file in target_files:
        pipeline.prepare_target_data(
            target_file=target_file,
            background_file=config.background_file,
            wavelength=config.wavelength,
            q_min_crop=config.q_min_crop,
            q_max_crop=config.q_max_crop,
        )

        for cfg, name in config.protocols:
            output_name = f"{target_file.split('.')[0]}_protocol_{name}.pkl"
            pipeline.run_experiment_protocol(
                n_trials=config.n_trials,
                protocol_name=name,
                save_to=output_name,
                **cfg
            )
            generated_pickles.append(os.path.join(layout.predictions_dir, output_name))

    layout.write_metadata({"predictions_dir": layout.predictions_dir})
    layout.write_metrics(
        {
            "target_files_processed": len(target_files),
            "protocol_count": len(config.protocols),
            "expected_prediction_pickles": len(generated_pickles),
        }
    )

if __name__ == "__main__":
    main()
