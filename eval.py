#!/usr/bin/env python

from train import *
from torchist import reduce_histogramdd, normalize, marginalize
from torchist.metrics import entropy, kl_divergence, em_distance


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('network', help='network file (PTH)')
    parser.add_argument('samples', help='samples file (H5)')

    parser.add_argument('-indices', nargs=2, type=int, default=(0, 1), help='indices range')
    parser.add_argument('-masks', nargs='+', default=['=1', '=2'], help='marginalization masks')
    parser.add_argument('-filter', default=None, help='mask filter')

    parser.add_argument('-bs', type=int, default=2 ** 12, help='batch size')
    parser.add_argument('-steps', type=int, default=2 ** 14, help='number of steps')
    parser.add_argument('-burn', type=int, default=2 ** 8, help='burning steps')
    parser.add_argument('-groupby', type=int, default=2 ** 8, help='sample group size')
    parser.add_argument('-sigma', type=float, default=2e-2, help='relative standard deviation')

    parser.add_argument('-bins', type=int, default=100, help='number of bins')
    parser.add_argument('-mcmc-limit', type=int, default=int(1e7), help='MCMC size limit')
    parser.add_argument('-emd-limit', type=int, default=int(1e4), help='EMD size limit')

    parser.add_argument('-clean', default=False, action='store_true')

    parser.add_argument('-accuracy', default=False, action='store_true')
    parser.add_argument('-calibration', default=False, action='store_true')
    parser.add_argument('-consistency', default=False, action='store_true')

    parser.add_argument('-classify', default=False, action='store_true')

    parser.add_argument('-o', '--output', default='products/results/out.csv', help='output file (CSV)')

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Output
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Settings
    settings = load_settings(args.network.replace('.pth', '.json'))
    settings['weights'] = args.network
    settings['samples'] = args.samples

    # Simulator & Model
    simulator, dataset, model, adversary = build_instance(settings)
    model.eval()

    low, high = simulator.low, simulator.high
    device = low.device

    # Masks
    theta_size = low.numel()

    if type(model) in [amnre.NRE, amnre.NPE]:
        masks = torch.tensor([[True] * theta_size])
    else:
        masks = amnre.list2masks(args.masks, theta_size, args.filter)

    # Samples
    for idx in tqdm(range(*args.indices)):
        theta_star, x_star = dataset[idx]

        if theta_star is not None:
            index_star = (theta_star - low) / (high - low)
            index_star = (args.bins * index_star).long()
            index_star = index_star.clip(max=args.bins - 1)
            index_star = index_star.cpu()

        ## Ground truth
        if args.accuracy and simulator.tractable:
            pthfile = args.samples.replace('.h5', f'_{idx}.pth')

            if not os.path.exists(pthfile):
                sampler = amnre.LESampler(
                    simulator.log_prob,
                    simulator.prior,
                    x_star,
                    batch_size=args.bs,
                    sigma=args.sigma * (high - low),
                )

                samples = sampler(args.steps, burn=args.burn, groupby=args.groupby)
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

        ## Surrogates
        metrics = []
        hists = {}
        divergences = {}

        y_star = model.embedding(x_star[None])[0]

        for mask in masks:
            textmask = amnre.mask2str(mask)

            if type(model) in [amnre.NRE, amnre.NPE]:
                ne = model
            else:
                ne = model[mask]
                if ne is None:
                    continue

            ### Hist
            numel = args.bins ** torch.count_nonzero(mask).item()

            if type(ne) is amnre.NPE:
                leakage = True
                sampler = amnre.PESampler(
                    ne,
                    y_star,
                    batch_size=args.bs,
                )
            else:
                leakage = False
                sampler = amnre.RESampler(
                    ne,
                    simulator.masked_prior(mask),
                    y_star,
                    batch_size=args.bs,
                    sigma=args.sigma * (high[mask] - low[mask]),
                )

            if numel > args.mcmc_limit:
                samples = sampler(args.steps, burn=args.burn, groupby=args.groupby)
                hist = reduce_histogramdd(
                    samples, args.bins,
                    low, high,
                    bounded=not leakage,
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

                if numel <= args.emd_limit:
                    metrics[-1]['emd_truth'] = em_distance(target, hist).item()
                else:
                    metrics[-1]['emd_truth'] = None

                del target

            #### Calibration
            if args.calibration and theta_star is not None:
                p = hist[tuple(index_star[mask])]

                if hist.is_sparse:
                    pdf = hist.values()
                else:
                    pdf = hist.view(-1)

                metrics[-1]['percentile'] = 1. - pdf[pdf >= p].sum().item()

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

                    divergences[textmask][key] = em_distance(p, q).item()
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
        if type(model) in [amnre.NPE, amnre.MNPE]:
            model.ratio()

        dataset = amnre.LTEDataset(dataset)
        length = len(dataset)

        with h5py.File(args.output.replace('.csv', '.h5'), 'w') as f:
            for mask in masks:
                textmask = amnre.mask2str(mask)

                f.create_dataset(textmask, (length * 2, 3))

            i = 0
            for theta, theta_prime, x in dataset:
                j, k = i + len(x), i + 2 * len(x)

                y = model.embedding(x)
                adv_y = adversary.embedding(x)

                for mask in masks:
                    textmask = amnre.mask2str(mask)

                    if type(model) in [amnre.NRE, amnre.NPE]:
                        nre = model
                        adv_nre = adversary
                    else:
                        nre = model[mask]
                        adv_nre = adversary[mask]

                        if nre is None:
                            continue

                    adv_theta = theta if settings['inverse'] else theta_prime
                    adv_ratio = adv_nre(adv_theta[..., mask], adv_y)

                    pred = nre(theta[..., mask], y).sigmoid().cpu().numpy()

                    if settings['inverse'] and adv_ratio is not None:
                        weight = (-adv_ratio).exp().cpu().numpy()
                    else:
                        weight = np.ones_like(pred)

                    f[textmask][i:j] = np.stack([np.ones_like(pred), pred, weight], axis=-1)

                    pred = nre(theta_prime[..., mask], y).sigmoid().cpu().numpy()

                    if not settings['inverse'] and adv_ratio is not None:
                        weight = adv_ratio.exp().cpu().numpy()
                    else:
                        weight = np.ones_like(pred)

                    f[textmask][j:k] = np.stack([np.zeros_like(pred), pred, weight], axis=-1)

                i = k
