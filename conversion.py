import pickle
from math import ceil

import numpy
import torch

from model_vc import Generator


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return numpy.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad


device = 'cpu'
G = Generator(32, 256, 512, 32).eval().to(device)

g_checkpoint = torch.load('autovc.ckpt', map_location=torch.device('cpu'))
G.load_state_dict(g_checkpoint['model'])

metadata = pickle.load(open('metadata.pkl', "rb"))


spect_vc = []

for sbmt_i in metadata:

    x_org = sbmt_i[2]
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[numpy.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][numpy.newaxis, :]).to(device)

    for sbmt_j in metadata:

        emb_trg = torch.from_numpy(sbmt_j[1][numpy.newaxis, :]).to(device)

        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)

        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

        spect_vc.append(('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg))


with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)
