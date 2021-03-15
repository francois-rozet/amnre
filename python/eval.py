#!/usr/bin/env python

from train import *
from histograms import *


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('settings', help='settings JSON file')
    parser.add_argument('sample', help='sample JSON file')

    parser.add_argument('-weights', help='weights file')

    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')
    parser.add_argument('-sigma', type=float, default=0.1, help='sigma')
    parser.add_argument('-start', type=int, default=2 ** 6, help='start sample')
    parser.add_argument('-stop', type=int, default=2 ** 14, help='end sample')
    parser.add_argument('-groupby', type=int, default=2 ** 8, help='sample group size')

    parser.add_argument('-bins', type=int, default=100, help='number of bins')
    parser.add_argument('-limit', type=int, default=int(1e7), help='histogram size limit before switching to MCMC')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Settings
    with open(args.settings) as f:
        settings = json.load(f)

    # Simulator & Model
    simulator, model = instance(settings)

    simulator.to(device)
    model.to(device)

    if args.weights is None:
        args.weights = args.settings.replace('.json', '.pth')

    model.load_state_dict(torch.load(args.weights, map_location=device))

    model.eval()

    # Truth sample
    with open(args.sample) as f:
        sample = json.load(f)

    theta_star = torch.tensor(sample['theta'])
    x_star = torch.tensor(sample['x'], device=device)

    # Histograms

    ## Ground Truth
    filename = args.sample.replace('.json', '.pdf')

    if not os.path.exists(filename):
        sampler = TractableSampler(
            simulator,
            x_star,
            batch_size=args.batch_size,
            sigma=args.sigma,
        )

        low, high = simulator.low, simulator.high

        samples = tqdm(sampler(args.start, args.stop, groupby=args.groupby))
        hist = reduce_histogramdd(samples, args.bins, low, high, bounded=True, sparse=True, device='cpu')
        hist = normalize(hist)

        pairs = get_pairs(hist)
        labels = [
            f'$\\theta_{{{i}}}$'
            for i in range(1, len(theta_star) + 1)
        ]

        fig = corner(pairs, low.cpu(), high.cpu(), truth=theta_star, labels=labels)
        plt.savefig(filename)
        plt.close()

        del hist, pairs

    ## MNRE
    filename = args.settings.replace('.json', '') + '_'
    filename += os.path.basename(args.sample).replace('.json', '') + '_'
    filename += '{}.pdf'

    with torch.no_grad():
        z_star = model.encoder(x_star.unsqueeze(0)).squeeze(0)

        for i, (mask, nre) in enumerate(model):
            low, high = simulator.low[mask], simulator.high[mask]

            sampler = RESampler(
                nre,
                simulator.subprior(mask),
                z_star,
                batch_size=args.batch_size,
                sigma=args.sigma,
            )

            size = args.bins ** mask.sum().item()
            if size > args.limit:
                samples = tqdm(sampler(args.start, args.stop, groupby=args.groupby))
                hist = reduce_histogramdd(samples, args.bins, low, high, bounded=True, sparse=True, device='cpu')
                hist = normalize(hist)
            else:
                hist = sampler.histogram(args.bins, low, high)

            pairs = get_pairs(hist)
            labels = [
                f'$\\theta_{{{i}}}$'
                for (i, b) in enumerate(mask, 1) if b
            ]

            fig = corner(pairs, low.cpu(), high.cpu(), truth=theta_star[mask], labels=labels)
            plt.savefig(filename.format(''.join(map(str, map(int, mask)))))
            plt.close()

            del hist, pairs
