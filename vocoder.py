import pickle
# from multiprocessing import Pool
from os import path

# import librosa
import soundfile
import torch

from synthesis import build_model, wavegen

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cpu")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth",
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])


def output(spect):
    name = spect[0]
    c = spect[1]
    filename = path.join(path.dirname(__file__), 'outs', name + '.wav')
    print(name, filename)
    waveform = wavegen(model, c=c)
    # librosa.output.write_wav(name+'.wav', waveform, sr=16000)
    soundfile.write(filename, waveform, samplerate=16000)


for spect in spect_vc:
    output(spect)
    # name = spect[0]
    # c = spect[1]
    # filename = path.join(path.dirname(__file__), name + '.csv')
    # print(name, filename)
    # waveform = wavegen(model, c=c)
    # # librosa.output.write_wav(name+'.wav', waveform, sr=16000)
    # soundfile.write(filename, waveform, samplerate=16000)


# if __name__ == '__main__':
#     p = Pool(6)
#     res = p.map(output, spect_vc)
#     print(res)
