


if __name__ == "__main__":
    import argparse
    import os
    import logging
    # suppresses warning by torcheval metrics for clients missing some classes
    # which is expected given Y-space mismatch does exist
    logging.getLogger().setLevel(logging.ERROR)

    from util import persist_run_impl
    from cover.train import CoverTrainer
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=105)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("-d", "--description", type=str, default="cover_split_by_wilderness")
    parser.add_argument("--ee-dim", type=int, default=8)
    parser.add_argument("--federate", action=argparse.BooleanOptionalAction)
    parser.add_argument("--personalize", action=argparse.BooleanOptionalAction)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-consistent-target-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-graph-and-embeddings", action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, default='cpu', choices=['mps', 'cpu', 'cuda'])
    parser.add_argument("--experiment", type=str, default='cover', choices=['cover'])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-range", type=str, default='') # example: --seed-range 1:10
    parser.add_argument("--model", type=str, required=True, default='model')

    args = parser.parse_args()

    if args.federate is None:
        print('Running both --federate and --no-federate as comparison.')
        args.federate = [True, False]
    else:
        print(f'Running only as {"federate" if args.federate else "no-federate"}.')
        args.federate = [args.federate]
    if True in args.federate and args.personalize is None:
            print('Running both --personalize and --no-personalize as comparison (for --federate runs only)')
            personalize = [True, False]
    else:
        personalize = [args.personalize]

    basepath = f'{args.experiment}/log/'
    basepath += str(len(next(os.walk(basepath))[1])+1).zfill(2)
    persist_run_impl(basepath)

    for fl_option in args.federate:
        for p_option in personalize:
            # print(f'starting fl_option={fl_option} and personalize={p_option}')
            if fl_option is False and p_option is True:
                if len(personalize) > 1:
                    # print('skipping as likley a duplicate run...')
                    continue
            same_yspace = args.force_consistent_target_space
            if fl_option and not p_option and not args.force_consistent_target_space:
                print('WARNING. With option --federate it is impossible to not --personalize '
                      'without also --force-consistent-target-space.'
                      'Otherwise output layer params will be mismatched and cannot be federated.\n'
                      'Overriding with --force-consistent-target-space set as True.')
                same_yspace = True

            name = 'personalized' if p_option else ''
            name += 'federated' if fl_option else 'orphaned'
            name += 'same_yspace' if same_yspace else ''

            if args.seed_range:
                sr = args.seed_range.split(':')
                seeds = range(int(sr[0]), int(sr[-1])+1)
                print(f'Running for seeds {list(seeds)}...')
            else:
                seeds = [args.seed]

            test_metrics = list()

            for i, seed in enumerate(seeds):
                print(f'Loading {name} trainer for seed {seed}...({i+1}/{len(seeds)})')
                trainer = CoverTrainer(run_description=f'{name}_{args.description}_{args.model}_{seed}', 
                                       batches=args.batches, 
                                       ee_dim=args.ee_dim,
                                       federate=fl_option,
                                       drop_last=args.drop_last,
                                       seed=seed,
                                       force_consistent_target_space=same_yspace,
                                       log_metadata=args.log_graph_and_embeddings,
                                       personalize=p_option,
                                       device=args.device,
                                       basepath=basepath + f'/seed_{seed}/',
                                       model_filename=args.model,
                                       name=args.experiment)
                print(f'Starting {name} training...')
                trainer.train(args.epochs)
                print(f'Done {name} training. Starting testing....')
                test_metric = trainer.test()
                test_metric.update(seed=seed)
                test_metrics.append(test_metric)
                print(f'Done {name} testing.')

            if len(test_metrics) > 1:
                import pandas as pd
                # aggregate into single summary
                df = pd.json_normalize(test_metrics)
                # select_dtypes() to drop 'confusion matrix' from aggregation
                # so we can cast float to :.4f. Otherwise, can remove.
                df.select_dtypes(exclude=object).T.reset_index().to_csv(
                    f'{basepath}/{name}_test_metrics.csv', 
                    float_format="%.4f", index=False, header=True)
                df.drop('seed', axis=1).describe().T.to_csv(
                    f'{basepath}/{name}_test_metrics_summary.csv', 
                    float_format="%.4f", index=True)
