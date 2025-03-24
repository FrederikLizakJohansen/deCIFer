from bin.experimental_pipeline import DeciferPipeline
from bin.train import TrainConfig # Needed
import argparse

def main(args):
    
    # Create pipeline
    pipeline = DeciferPipeline(
        model_path = args.model_path,
        zip_path = args.zip_path
    )

    """ Crystalline """

    # Ceria
    pipeline.setup_folder(f"crystalline_CeO2_protocol_{args.suffix}")
    target_files = ["crystalline_CeO2_BM31.xye"]
    background_file = None

    wavelength=0.25448
    q_min_crop=1.5
    q_max_crop=8.0

    protocols = [
        [{}, "none"],
        [{"composition": "CexO2x", "composition_ranges": {"Ce": (1,4)}}, "CexO2x"],
        [{"composition": "CexO2x", "composition_ranges": {"Ce": (1,4)}, "spacegroup": "Fm-3m_sg"}, "CexO2x_sg"],
    ]

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

    # Silicon
    pipeline.setup_folder(f"crystalline_Si_protocol_{args.suffix}")
    target_files = ["Si_Mythen.xye"]
    background_file = None

    wavelength=0.825008
    q_min_crop=1.8
    q_max_crop=8

    protocols = [
        [{}, "none"],
        [{"composition": "Six", "composition_ranges": {"Si": (1,8)}}, "Six"],
        [{"composition": "Si8"}, "Si8"],
        [{"composition": "Six", "composition_ranges": {"Si": (1,4)}, "crystal_systems": [4,5,6,7]}, "Six_crystal"],
        [{"composition": "Si1", "crystal_systems": [7]}, "Si1_cubic"],
    ]

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

    # Fe2O3
    pipeline.setup_folder(f"crystalline_Fe2O3_protocol_{args.suffix}")
    target_files = ["AFS012d_a850C.xy"]
    background_file = None

    wavelength=1.5406
    q_min_crop=1
    q_max_crop=8

    protocols = [
        [{}, "none"],
        [{"composition": "Fe12O18"}, "Fe12O18"],
        [{"composition": "Fe2xO3x", "composition_ranges": {"Fe": (1,4)}, "spacegroup": "R-3c_sg"}, "Fe2xO3x_sg"],
        [{"composition": "Fe12O18", "spacegroup": "R-3c_sg"}, "Fe12O18_sg"],
        [{"composition": "Six", "composition_ranges": {"Si": (1,4)}}, "Six"],
        [{"composition": "Six", "composition_ranges": {"Si": (1,4)}, "spacegroup": "Fd-3m_sg"}, "Six_sg"],
    ]

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
