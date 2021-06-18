#!/usr/bin/env python

from train import *
from torchist import reduce_histogramdd, normalize, marginalize
from torchist.metrics import entropy, kl_divergence, w_distance


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('settings', help='settings file (JSON)')
    parser.add_argument('samples', help='samples file (H5)')

    parser.add_argument('-indices', nargs=2, type=int, default=(0, 1), help='indices range')
    parser.add_argument('-masks', nargs='+', default=['=1', '=2'], help='marginalization masks')

    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')
    parser.add_argument('-sigma', type=float, default=2e-2, help='relative standard deviation')
    parser.add_argument('-start', type=int, default=2 ** 6, help='start sample')
    parser.add_argument('-stop', type=int, default=2 ** 14, help='end sample')
    parser.add_argument('-groupby', type=int, default=2 ** 8, help='sample group size')

    parser.add_argument('-bins', type=int, default=100, help='number of bins')
    parser.add_argument('-mcmc-limit', type=int, default=int(1e7), help='MCMC size limit')
    parser.add_argument('-wd-limit', type=int, default=int(1e4), help='Wasserstein distance size limit')

    parser.add_argument('-clean', default=False, action='store_true')

    parser.add_argument('-accuracy', default=False, action='store_true')
    parser.add_argument('-coverage', default=False, action='store_true')
    parser.add_argument('-consistency', default=False, action='store_true')

    parser.add_argument('-classify', default=False, action='store_true')

    parser.add_argument('-o', '--output', default=None, help='output file (CSV)')

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Output
    if args.output is None:
        args.output = args.settings.replace('.json', '_')
        args.output += os.path.basename(args.samples).replace('.h5', '.csv')
    elif os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Settings
    settings = load_settings(args.settings)
    settings['samples'] = args.samples

    # Simulator & Model
    simulator, dataset, model = build_instance(settings)
    model.eval()

    low, high = simulator.low, simulator.high
    device = low.device

    # Masks
    theta_size = low.numel()

    if type(model) is amsi.NRE:
        masks = torch.tensor([[True] * theta_size])
    else:
        masks = amsi.list2masks(args.masks, theta_size)

    # Samples
    for idx in tqdm(range(*args.indices)):
        theta_star, x_star = dataset[idx]

        if theta_star is not None:
            index_star = (theta_star - low) / (high - low)
            index_star = (args.bins * index_star).long()
            index_star = index_star.clip(max=args.bins - 1)

            theta_star, index_star = theta_star.cpu(), index_star.cpu()

        ## Ground truth
        if args.accuracy and simulator.tractable:
            pthfile = args.samples.replace('.h5', f'_{idx}.pth')

            if not os.path.exists(pthfile):
                sampler = amsi.TractableSampler(
                    simulator,
                    x_star,
                    batch_size=args.batch_size,
                    sigma=args.sigma * (high - low),
                )

                samples = sampler(args.start, args.stop, groupby=args.groupby)
                truth = reduce_histogramdd(
                    samples, args.bins,
                    low, high,
                    bounded=True,
                    sparse=True,
                    device='cpu',
                ).coalesce()

                truth, _ = normalize(truth)

                mask = torch.tensor([True] * theta_size)
                torch.save((mask, truth), pthfile)
            else:
                _, truth = torch.load(pthfile)
                truth._coalesced_(True)
        else:
            truth = None

        ## MNRE
        metrics = []
        hists = {}
        divergences = {}

        z_star = model.encoder(x_star)

        for mask in masks:
            textmask = amsi.mask2str(mask)

            if type(model) is amsi.NRE:
                nre = model
            else:
                nre = model[mask]
                if nre is None:
                    continue

            ### Hist
            numel = args.bins ** torch.count_nonzero(mask).item()

            sampler = amsi.RESampler(
                nre,
                simulator.masked_prior(mask),
                z_star,
                batch_size=args.batch_size,
                sigma=args.sigma * (high[mask] - low[mask]),
            )

            if numel > args.mcmc_limit:
                samples = sampler(args.start, args.stop, groupby=args.groupby)
                hist = reduce_histogramdd(
                    samples, args.bins,
                    low, high,
                    bounded=True,
                    sparse=True,
                    device='cpu',
                ).coalesce()
            else:
                hist = sampler.histogram(args.bins, low[mask], high[mask])
                hist = torch.nan_to_num(hist)

            ### Metrics
            hist, total = normalize(hist)

            metrics.append({
                'index': idx,
                'mask': textmask,
                'total_probability': total.item(),
                'entropy': entropy(hist).item(),
            })

            #### Accuracy w.r.t. ground truth
            if truth is not None:
                dims = torch.arange(len(mask))[~mask]
                target = marginalize(truth, dim=dims.tolist())

                if not hist.is_sparse:
                    target = target.to_dense()

                target = target.to(hist)

                metrics[-1]['entropy_truth'] = entropy(target).item()
                metrics[-1]['kl_truth'] = kl_divergence(target, hist).item()

                if numel <= args.wd_limit:
                    metrics[-1]['wd_truth'] = w_distance(target, hist).item()
                else:
                    metrics[-1]['wd_truth'] = None

                del target

            #### Coverage
            if args.coverage and theta_star is not None:
                p = hist[tuple(index_star[mask])]

                if hist.is_sparse:
                    pdf = hist.values()
                else:
                    pdf = hist.view(-1)

                metrics[-1]['quantile'] = pdf[pdf >= p].sum().item()

            #### Consistency
            hist = hist.cpu()

            if args.consistency and not hist.is_sparse:
                divergences[textmask] = {textmask: None}

                for key, (m, h) in hists.items():
                    common = torch.logical_and(mask, m)

                    if torch.all(~common):
                        divergences[textmask][key] = None
                        divergences[key][textmask] = None
                        continue

                    dims = mask.cumsum(0)[common] - 1
                    p = marginalize(hist, dim=dims.tolist(), keep=True)

                    dims = m.cumsum(0)[common] - 1
                    q = marginalize(h, dim=dims.tolist(), keep=True)

                    divergences[textmask][key] = w_distance(p, q).item()
                    divergences[key][textmask] = divergences[textmask][key]

                hists[textmask] = mask, hist

            ### Export histogram
            if not args.clean:
                torch.save((mask, hist), args.output.replace('.csv', f'_{idx}_{textmask}.pth'))

        ## Append metrics
        df = pd.DataFrame(metrics)
        df.to_csv(
            args.output,
            index=False,
            mode='a',
            header=not os.path.exists(args.output),
        )

        ## Export consistency
        if args.consistency:
            df = pd.DataFrame(divergences)
            df.to_csv(args.output.replace('.csv', f'_{idx}.csv'))

    # Classification
    if args.classify:
        if settings['adversary'] is None:
            adversary = Dummy()
        else:
            _, _, adversary = build_instance(load_settings(settings['adversary']))

        dataset = amsi.LTEDataset(dataset)
        length = len(dataset)

        with h5py.File(args.output.replace('.csv', '.h5'), 'w') as f:
            for mask in masks:
                textmask = amsi.mask2str(mask)

                f.create_dataset(textmask, (length * 2, 3))

            i = 0
            for theta, theta_prime, x in dataset:
                j, k = i + len(x), i + 2 * len(x)

                z = model.encoder(x)
                adv_z = adversary.encoder(x)

                for mask in masks:
                    textmask = amsi.mask2str(mask)

                    if type(model) is amsi.NRE:
                        nre = model
                        adv_nre = adversary
                    else:
                        nre = model[mask]
                        adv_nre = adversary[mask]

                        if nre is None:
                            continue

                    pred = nre(theta, z).sigmoid().cpu().numpy()
                    f[textmask][i:j] = np.stack([np.ones_like(pred), pred, np.ones_like(pred)], axis=-1)

                    pred = nre(theta_prime, z).sigmoid().cpu().numpy()

                    weight = adv_nre(theta_prime, adv_z)
                    if weight is None:
                        weight = np.ones_like(pred)
                    else:
                        weight = weight.exp().cpu().numpy()

                    f[textmask][j:k] = np.stack([np.zeros_like(pred), pred, weight], axis=-1)

                i = k
