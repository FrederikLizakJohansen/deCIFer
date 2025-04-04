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

    ## Ceria
    # pipeline.setup_folder(f"crystalline_CeO2_protocol_{args.suffix}")
    # target_files = ["crystalline_CeO2_BM31.xye"]
    # background_file = None

    # wavelength=0.25448
    # q_min_crop=1.5
    # q_max_crop=8.0

    # protocols = [
    #     [{}, "none"],
    #     [{'spacegroup': "Fm-3m_sg"}, "Fm-3m"],
    #     [{'crystal_systems':[7]}, "Cubic"],
    #     [{"composition": "Ce1O2"}, "Ce1O2"],
    #     [{"composition": "Ce2O4"}, "Ce2O4"],
    #     [{"composition": "Ce4O8"}, "Ce4O8"],
    #     [{"composition": "Ce4O8", "spacegroup": "Fm-3m_sg"}, "Ce4O8_Fm-3m"],
    # ]

    # for target_file in target_files[:args.debug_max]:
    #     pipeline.prepare_target_data(
    #         target_file=target_file,
    #         background_file=background_file,
    #         wavelength=wavelength,
    #         q_min_crop=q_min_crop,
    #         q_max_crop=q_max_crop,
    #     )

    #    for cfg, name in protocols:
    #         pipeline.run_experiment_protocol(
    #             n_trials=args.n_trials,
    #             protocol_name=name,
    #             save_to=f"{target_file.split('.')[0]}_protocol_{name}.pkl", 
    #             **cfg
    #         )

    # Silicon
    pipeline.setup_folder(f"crystalline_Si_protocol_{args.suffix}")
    target_files = ["Si_Mythen.xye"]
    background_file = None

    wavelength=0.825008
    q_min_crop=1.8
    q_max_crop=8

    protocols = [
        [{}, "none"],
        [{'spacegroup': "Fm-3m_sg"}, "Fm-3m"],
        [{'spacegroup': "Fd-3m_sg"}, "Fd-3m"],
        [{'crystal_systems':[7]}, "Cubic"],
        [{"composition": "Si4"}, "Si4"],
        [{"composition": "Si8"}, "Si8"],
        [{"composition": "Si4", "spacegroup": "Fm-3m_sg"}, "Si4_Fm-3m"],
        [{"composition": "Si8", "spacegroup": "Fd-3m_sg"}, "Si8_Fd-3m"],
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
    q_min_crop=0.5
    q_max_crop=8

    protocols = [
        [{}, "none"],
        [{'crystal_systems':[1]}, "Trigonal"],
        [{'crystal_systems':[7]}, "Cubic"],
        [{'crystal_systems':[3]}, "Orthorhombic"],
        [{'spacegroup': "R-3c_sg"}, "R-3c"],
        [{'spacegroup': "Ia-3_sg"}, "Ia-3"],
        [{'spacegroup': "Pna2_1_sg"}, "Pna2_1"],
        [{"composition": "Fe12O18"}, "Fe12O18"],
        [{"composition": "Fe32O48"}, "Fe32O48"],
        [{"composition": "Fe16O24"}, "Fe16O24"],
        [{"composition": "Fe12O18", "spacegroup": "R-3c_sg"}, "Fe12O18_R-3c"],
        [{"composition": "Fe32O48", "spacegroup": "Ia-3_sg"}, "Fe12O18_Ia-3"],
        [{"composition": "Fe16O24", "spacegroup": "Pna2_1_sg"}, "Fe12O18_Pna2_1"],
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
    
    """ CeO2 particles """

    # Ceria
    pipeline.setup_folder(f"particles_CeO2_protocol_{args.suffix}")
    target_files = [
        "Hydrolyse_ID5_20min_3-56_boro_0p8.xy",
        "Hydrolyse_ID6_20min_3-56_boro_0p8.xy",
        "Hydrolyse_ID8_20min_3-56_boro_0p8.xy",
        "Hydrolyse_ID10_20min_3-56_boro_0p8.xy",
    ]
    background_file = "boroglass_0p8_empty_VCT_72h.xy"

    wavelength=0.5594075
    q_min_crop=1.5
    q_max_crop=8.0

    protocols = [
        [{}, "none"],
        [{'spacegroup': "Fm-3m_sg"}, "Fm-3m"],
        [{'crystal_systems':[7]}, "Cubic"],
        [{"composition": "Ce1O2"}, "Ce1O2"],
        [{"composition": "Ce2O4"}, "Ce2O4"],
        [{"composition": "Ce4O8"}, "Ce4O8"],
        [{"composition": "Ce4O8", "spacegroup": "Fm-3m_sg"}, "Ce4O8_Fm-3m"],
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
