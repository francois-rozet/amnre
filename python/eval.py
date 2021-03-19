#!/usr/bin/env python

from train import *
from histograms import *


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('settings', help='settings file (JSON)')
    parser.add_argument('truth', help='truth sample file (JSON)')

    parser.add_argument('-plots', default=False, action='store_true')

    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')
    parser.add_argument('-sigma', type=float, default=0.1, help='sigma')
    parser.add_argument('-start', type=int, default=2 ** 6, help='start sample')
    parser.add_argument('-stop', type=int, default=2 ** 15, help='end sample')
    parser.add_argument('-groupby', type=int, default=2 ** 8, help='sample group size')

    parser.add_argument('-bins', type=int, default=100, help='number of bins')
    parser.add_argument('-limit', type=int, default=int(1e7), help='histogram size limit')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Settings
    with open(args.settings) as f:
        settings = json.load(f)

    settings['weights'] = args.settings.replace('.json', '.pth')

    # Simulator & Model
    simulator, model = build_instance(settings)

    simulator.to(device)
    model.to(device)
    model.eval()

    # Ground Truth

    ## Sample
    with open(args.truth) as f:
        truth = json.load(f)

    theta_star = torch.tensor(truth['theta'])
    x_star = torch.tensor(truth['x'], device=device)

    low, high = simulator.low, simulator.high

    ## Hist
    filename = args.truth.replace('.json', '.pth')

    if not os.path.exists(filename):
        sampler = acsi.TractableSampler(
            simulator,
            x_star,
            batch_size=args.batch_size,
            sigma=args.sigma,
        )

        samples = sampler(args.start, args.stop, groupby=args.groupby)
        truth = sparse_histogram(samples, args.bins, low, high)

        torch.save(truth, filename)
    else:
        truth = torch.load(filename).coalesce()

    truth = normalize(truth)

    ## Plot
    if args.plots:
        pairs = get_pairs(truth)
        labels = get_labels([True] * truth.dim())

        corner(
            pairs, low.cpu(), high.cpu(),
            labels=labels, truth=theta_star,
            filename=args.truth.replace('.json', '.pdf'),
        )

        del pairs, labels

    # MNRE
    filename = args.settings.replace('.json', '_')
    filename += os.path.basename(args.truth).replace('.json', '')

    hists = []
    measures = []
    divergences = []

    with torch.no_grad():
        z_star = model.encoder(x_star.unsqueeze(0)).squeeze(0)

        for mask, nre in model:
            size = args.bins ** torch.count_nonzero(mask)
            if size > args.limit:
                continue

            ## Hist
            sampler = acsi.RESampler(
                nre,
                simulator.subprior(mask),
                z_star,
                batch_size=args.batch_size,
                sigma=args.sigma,
            )

            hist = sampler.histogram(args.bins, low[mask], high[mask])

            ## Plot
            textmask = ''.join(map(str, map(int, mask)))

            if args.plots:
                pairs = get_pairs(hist)
                labels = get_labels(mask)

                fig = corner(
                    pairs, low[mask].cpu(), high[mask].cpu(),
                    labels=labels, truth=theta_star[mask],
                    filename=filename + f'_{textmask}.pdf',
                )

                del pairs, labels

            ## Accuracy
            dims = torch.arange(len(mask))[mask]
            target = marginalize(truth, dim=dims.tolist(), keep=True).to_dense().to(hist)

            tot = marginalize(hist, dim=[], keep=True)
            hist = normalize(hist)

            measures.append({
                'mask': textmask,
                'total_probability': tot.item(),
                'entropy': entropy(hist).item(),
                'target_entropy': entropy(target).item(),
                'kl': kl_divergence(target, hist).item(),
            })

            del target

            ## Consistence
            divergences.append([])

            for i, (m, h) in enumerate(hists):
                common = torch.logical_and(mask, m)

                if torch.all(~common):
                    divergences.append(0.)
                    continue

                dims = mask.cumsum(0)[common] - 1
                p = marginalize(hist, dim=dims.tolist(), keep=True)

                dims = m.cumsum(0)[common] - 1
                q = marginalize(h, dim=dims.tolist(), keep=True)

                divergences[i].append(kl_divergence(q, p).item())
                divergences[-1].append(kl_divergence(p, q).item())

            divergences[-1].append(0.)

            hists.append((mask, hist))

    # Save
    measures = pd.DataFrame(measures)
    measures.to_csv(filename + '.csv', index=False)

    divergences = np.asarray(divergences)
    np.savetxt(filename + '.txt', divergences, fmt='%.6f')
