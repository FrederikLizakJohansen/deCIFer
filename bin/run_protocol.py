from bin.experimental_pipeline import DeciferPipeline
from bin.train import TrainConfig # Needed
import argparse

def main(args):
    
    # Create pipeline
    pipeline = DeciferPipeline(
        model_path = args.model_path,
        zip_path = args.zip_path
    )

    # Ceria
    pipeline.setup_folder(f"particles_CeO2_protocol_{args.suffix}")
    
    protocols = [
        [{}, "none"],
        [{'spacegroup': "Fm-3m_sg"}, "Fm-3m"],
        [{'crystal_systems':[7]}, "Cubic"],
        [{"composition": "Ce1O2"}, "Ce1O2"],
        [{"composition": "Ce2O4"}, "Ce2O4"],
        [{"composition": "Ce4O8"}, "Ce4O8"],
        [{"composition": "Ce4O8", "spacegroup": "Fm-3m_sg"}, "Ce4O8_Fm-3m"],
    ]
    
    target_files = [
        "scan-4907_mean.xy",
        "scan-4911_mean.xy",
        "scan-4912_mean.xy",
        "scan-4919_mean.xy",
    ]
    background_file = "scan-4903_mean.xy"

    wavelength=None # Already in Q
    q_min_crop=1.5
    q_max_crop=8.0

    for target_file in target_files[:args.debug_max]:
        pipeline.prepare_target_data(
            target_file=target_file,
            background_file=background_file,
            wavelength=wavelength,
            q_min_crop=q_min_crop,
            q_max_crop=q_max_crop,
        )

        for cfg, name in protocols:
            pipeline.run_experiment_protocol(
                n_trials=args.n_trials,
                protocol_name=name,
                save_to=f"{target_file.split('.')[0]}_protocol_{name}.pkl", 
                **cfg
            )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-path", type=str, required=True)
    argparser.add_argument("--zip-path", type=str, required=True)
    argparser.add_argument("--debug-max", type=int, default=0)
    argparser.add_argument("--n-trials", type=int, default=25)
    argparser.add_argument("--suffix", type=str, default='default')
    args = argparser.parse_args()
    args.debug_max = None if args.debug_max == 0 else args.debug_max
    main(args)
