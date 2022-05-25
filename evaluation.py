from __future__ import print_function
import os
import pickle
from re import S

import numpy
from data import get_test_loader, get_transform
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE, order_sim
from collections import OrderedDict

import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from data import get_paths
from pycocotools.coco import COCO
from PIL import Image
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    with torch.no_grad():
        for i, (images, captions, lengths, ids) in enumerate(data_loader):
            # make sure val logger is used
            model.logger = val_logger
            
            #if i == 0:


            # compute the embeddings
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

            # preserve the embeddings by copying from gpu and converting to numpy
            for i, id in enumerate(ids):
                img_embs[id] = img_emb.data.cpu().numpy().copy()[i]
                cap_embs[id] = cap_emb.data.cpu().numpy().copy()[i]

            # measure accuracy and record loss
            model.forward_loss(img_emb, cap_emb)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(
                            i, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))
            del images, captions

    return img_embs, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join('vocab/',
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    if hasattr(opt,'use_bert') == True and opt.use_bert == True:
        vocab = None
        opt.vocab_size = None

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    dpath = os.path.join(opt.data_path,opt.data_name)
    roots, ids = get_paths(dpath,opt.data_name,opt.use_restval)
    ids = ids[split]
    ann_id = ids[65]
    json = roots[split]['cap']
    root = roots[split]['img']
    coco = COCO(json)
    img_id = coco.anns[ann_id]['image_id']
    path = coco.loadImgs(img_id)[0]['file_name']
    im = Image.open(os.path.join(root,path))
    #im.show()

    caption = coco.anns[ann_id]['caption']
    print(caption)
    print(path)

    #check_id = []
    check_id = [19830,19834,19831,18411,19833]

    for id in check_id:
        ann_id = ids[id]
        json = roots[split]['cap']
        root = roots[split]['img']
        coco = COCO(json)
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        im = Image.open(os.path.join(root,path))
        #im.show()

        caption = coco.anns[ann_id]['caption']
        print("caption id:",id)
        print(caption)
        print(path)
        print("-----")



    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure=opt.measure,
                         return_ranks=True,ii=i)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure=opt.measure,
                           return_ranks=True,ii=i)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False,ii=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """

    count_rank = []

    if npts is None:
        npts = int(images.shape[0] / 5)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if (ii*1000+index) == 3682:
                print("rank:",tmp)
        ranks[index] = rank
        top1[index] = inds[0]
        count_rank.append(rank)

        #if ii == None or rank != 0 or index % 2 == 0 or 1 == 1:
        if (ii*1000+index) != 3682:
            continue
        print("rank",rank)
        print("caption idx:",5*index+ii*5000)
        s = "["
        for j in range(5):
            s += str(math.floor(5*index+ii*5000+j))
            s += ","
        torank = 15
        for j in range(torank):
            s += str(math.floor(inds[j]+ii*5000))
            if j != (torank-1):
                s += ","
        s += "]"
        print(s)
        print("-----------------------")


    if ii != None:
        arr_1d = np.array(count_rank)
        #np.save('./i2t_resnetlstm_rank_'+str(ii),arr_1d)
        print("shape:",arr_1d.shape)
    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False,ii=None):

    diff_rank = []
    #diff_rank = [19592,18707,4303,19113,1302,11540,4583,21887,4776,13426,12435,14315,10711,23657,22724,7556,15576,12740,4197,6982,11155,16649,4302,23576,16311,16022,4285,20512,22388,14993]

    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    count_rank = []

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        #print("d.shape",d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            #print("index",index)
            #print("inds[i]",inds[i])
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

            if ii == None:
                continue
            #print("rank",ranks[5 * index + i])

            count_rank.append(ranks[5 * index + i])

            #if ranks[5 * index + i] != -1:
            if (ii*5000+index*5+i) not in diff_rank:
                continue

            
            print("rank",ranks[5 * index + i])
            print("caption idx:",ii*5000+index*5+i)
            s = "[" + str(ii*5000+index*5+i) + ","
            torank = 30
            for j in range(torank):
            #    print(j,":",math.floor(inds[i][j]*5+ii*5000))
                s += str(math.floor(inds[i][j]*5+ii*5000))
                if j != (torank-1):
                    s += ","
            s += "]"
            print(s)
            print("-----------------------")

    if ii != None:
        arr_1d = np.array(count_rank)
        #np.save('./rank_resnetlstm_'+str(ii), arr_1d)
        print("shape:",arr_1d.shape)
    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
