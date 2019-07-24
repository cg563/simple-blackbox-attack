import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import utils
import math
import random
import torch.nn.functional as F
import argparse
import os
import pdb

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--freq_dims', type=int, default=14, help='dimensionality of 2D frequency space')
parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
parser.add_argument('--stride', type=int, default=7, help='stride for block order')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
args = parser.parse_args()

def expand_vector(x, size):
    batch_size = x.size(0)
    x = x.view(-1, 3, size, size)
    z = torch.zeros(batch_size, 3, image_size, image_size)
    z[:, :, :size, :size] = x
    return z

def normalize(x):
    return utils.apply_normalization(x, 'imagenet')

def get_probs(model, x, y):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    probs = torch.index_select(torch.nn.Softmax()(output).data, 1, y)
    return torch.diag(probs)

def get_preds(model, x):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    _, preds = output.data.max(1)
    return preds

# runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
# (for targeted attack) <labels_batch>
def dct_attack_batch(model, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, order='rand', targeted=False, pixel_attack=False, log_every=1):
    batch_size = images_batch.size(0)
    image_size = images_batch.size(2)
    # sample a random ordering for coordinates independently per batch element
    if order == 'rand':
        indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
    elif order == 'diag':
        indices = utils.diagonal_order(image_size, 3)[:max_iters]
    elif order == 'strided':
        indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
    else:
        indices = utils.block_order(image_size, 3)[:max_iters]
    if order == 'rand':
        expand_dims = freq_dims
    else:
        expand_dims = image_size
    n_dims = 3 * expand_dims * expand_dims
    x = torch.zeros(batch_size, n_dims)
    # logging tensors
    probs = torch.zeros(batch_size, max_iters)
    succs = torch.zeros(batch_size, max_iters)
    queries = torch.zeros(batch_size, max_iters)
    l2_norms = torch.zeros(batch_size, max_iters)
    linf_norms = torch.zeros(batch_size, max_iters)
    prev_probs = get_probs(model, images_batch, labels_batch)
    preds = get_preds(model, images_batch)
    if pixel_attack:
        trans = lambda z: z
    else:
        trans = lambda z: utils.block_idct(z, block_size=image_size)
    remaining_indices = torch.arange(0, batch_size).long()
    for k in range(max_iters):
        dim = indices[k]
        expanded = (images_batch[remaining_indices] + trans(expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
        perturbation = trans(expand_vector(x, expand_dims))
        l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
        linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
        preds_next = get_preds(model, expanded)
        preds[remaining_indices] = preds_next
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        # check if all images are misclassified and stop early
        if remaining.sum() == 0:
            adv = (images_batch + trans(expand_vector(x, expand_dims))).clamp(0, 1)
            probs_k = get_probs(model, adv, labels_batch)
            probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
            succs[:, k:] = torch.ones(args.batch_size, max_iters - k)
            queries[:, k:] = torch.zeros(args.batch_size, max_iters - k)
            break
        remaining_indices = torch.arange(0, batch_size)[remaining].long()
        if k > 0:
            succs[:, k-1] = 1 - remaining
        diff = torch.zeros(remaining.sum(), n_dims)
        diff[:, dim] = epsilon
        left_vec = x[remaining_indices] - diff
        right_vec = x[remaining_indices] + diff
        # trying negative direction
        adv = (images_batch[remaining_indices] + trans(expand_vector(left_vec, expand_dims))).clamp(0, 1)
        left_probs = get_probs(model, adv, labels_batch[remaining_indices])
        queries_k = torch.zeros(batch_size)
        # increase query count for all images
        queries_k[remaining_indices] += 1
        if targeted:
            improved = left_probs.gt(prev_probs[remaining_indices])
        else:
            improved = left_probs.lt(prev_probs[remaining_indices])
        # only increase query count further by 1 for images that did not improve in adversarial loss
        if improved.sum() < remaining_indices.size(0):
            queries_k[remaining_indices[1-improved]] += 1
        # try positive directions
        adv = (images_batch[remaining_indices] + trans(expand_vector(right_vec, expand_dims))).clamp(0, 1)
        right_probs = get_probs(model, adv, labels_batch[remaining_indices])
        if targeted:
            right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
        else:
            right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
        probs_k = prev_probs.clone()
        # update x depending on which direction improved
        if improved.sum() > 0:
            left_indices = remaining_indices[improved]
            left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
            x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
            probs_k[left_indices] = left_probs[improved]
        if right_improved.sum() > 0:
            right_indices = remaining_indices[right_improved]
            right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
            x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
            probs_k[right_indices] = right_probs[right_improved]
        probs[:, k] = probs_k
        queries[:, k] = queries_k
        prev_probs = probs[:, k]
        if (k + 1) % log_every == 0 or k == max_iters - 1:
            print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                    k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
    expanded = (images_batch + trans(expand_vector(x, expand_dims))).clamp(0, 1)
    preds = get_preds(model, expanded)
    if targeted:
        remaining = preds.ne(labels_batch)
    else:
        remaining = preds.eq(labels_batch)
    succs[:, max_iters-1] = 1 - remaining
    return x, probs, succs, queries, l2_norms, linf_norms

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.exists(args.sampled_image_dir):
    os.mkdir(args.sampled_image_dir)

# load model and dataset
model = getattr(models, args.model)(pretrained=True).cuda()
model.eval()
if args.model.startswith('inception'):
    image_size = 299
    #testset = dset.ImageFolder(args.data_root + '/val', utils.INCEPTION_TRANSFORM)
else:
    image_size = 224
    #testset = dset.ImageFolder(args.data_root + '/val', utils.IMAGENET_TRANSFORM)

# load sampled images or sample new ones
# this is to ensure all attacks are run on the same set of correctly classified images
batchfile = '%s/images_%s_%d.pth' % (args.sampled_image_dir, args.model, args.num_runs)
if os.path.isfile(batchfile):
    checkpoint = torch.load(batchfile)
    images = checkpoint['images']
    labels = checkpoint['labels']
else:
    images = torch.zeros(args.num_runs, 3, image_size, image_size)
    labels = torch.zeros(args.num_runs).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model, images[idx], 'imagenet', batch_size=args.batch_size)
    torch.save({'images': images, 'labels': labels}, batchfile)

if args.order == 'rand':
    n_dims = 3 * args.freq_dims * args.freq_dims
else:
    n_dims = 3 * image_size * image_size
if args.num_iters > 0:
    max_iters = int(min(n_dims, args.num_iters))
else:
    max_iters = int(n_dims)
N = int(math.floor(float(args.num_runs) / float(args.batch_size)))
for i in range(N):
    upper = min((i + 1) * args.batch_size, args.num_runs)
    images_batch = images[(i * args.batch_size):upper]
    labels_batch = labels[(i * args.batch_size):upper]
    # replace true label with random target labels in case of targeted attack
    if args.targeted:
        labels_targeted = labels_batch.clone()
        while labels_targeted.eq(labels_batch).sum() > 0:
            labels_targeted = torch.floor(1000 * torch.rand(labels_batch.size())).long()
        labels_batch = labels_targeted
    x, probs, succs, queries, l2_norms, linf_norms = dct_attack_batch(
        model, images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, order=args.order,
        targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
    if i == 0:
        all_vecs = x
        all_probs = probs
        all_succs = succs
        all_queries = queries
        all_l2_norms = l2_norms
        all_linf_norms = linf_norms
    else:
        all_vecs = torch.cat([all_vecs, x], dim=0)
        all_probs = torch.cat([all_probs, probs], dim=0)
        all_succs = torch.cat([all_succs, succs], dim=0)
        all_queries = torch.cat([all_queries, queries], dim=0)
        all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
        all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
    if args.pixel_attack:
        prefix = 'pixel'
    else:
        prefix = 'dct'
    if args.targeted:
        prefix += '_targeted'
    savefile = '%s/%s_%s_%d_%d_%d_%.4f_%s%s.pth' % (
        args.result_dir, prefix, args.model, args.num_runs, args.num_iters, freq_dims, args.epsilon, args.order, args.save_suffix)
    torch.save({'original': images, 'vecs': all_vecs, 'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
                'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms}, savefile)
