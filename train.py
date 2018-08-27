# ===== modules ===== #
import glob, os
from librosa.core import load, stft
from tqdm import tqdm
import argparse
from random import shuffle
import numpy as np
import chainer
import chainer.links as L
from chainer.training import extensions
from network_spec import Spec_classification

# ===== config ===== #
SR = 16000
T_LEN = 10
N_FFT = 512             # 32ms
HOP_LEN = int(N_FFT/2)  # 16ms

# ===== utils ===== #
def rolling_spectrogram(spec, winLen=64):
    res = []
    for i in range(spec.shape[2]-winLen):
        res.append(spec[:,:,(i):(i+winLen)])
    return res

# ===== main ===== #
def main():
    # - Argparse - #
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="GPU or CPU, minus values mean CPU")
    parser.add_argument("--data0", type=str, default="../data/ok/LA-highpass-clean/",
                        help="Relative path to data directory whose labeld 0")
    parser.add_argument("--data1", type=str, default="../data/ok/LA-highpass-noisy/",
                        help="Relative path to data directory whose labeld 1")
    parser.add_argument("--epoch", "-e", type=int, default=10,
                        help="the number of epochs")
    parser.add_argument("--batchsize", "-b", type=int, default=1,
                        help="the number of data in minibatch")
    parser.add_argument("--winlen", type=int, default=32,
                        help="the number of spectrogram time bins")
    parser.add_argument("--result", type=str, default="result/",
                        help="Relative path to saving result directory")    
    args = parser.parse_args()

    # - Set model - #
    model = Spec_classification()
    if args.gpu >= 0:
        xp = chainer.cuda.cupy
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    else:
        xp = np
    
    # - Load data - #
    print("data loading...")
    d0 = glob.glob( os.path.join(args.data0, "*.wav") )
    d1 = glob.glob( os.path.join(args.data1, "*.wav") )
    
    train = []
    valid = []
    train_tmp = []
    label0 = np.array(0, dtype=np.int32)
    label1 = np.array(1, dtype=np.int32)
    print('load data whose label 0...')
    for i in tqdm( d0 ):
        tmp = load(i, sr=SR)[0]
        tmp = np.abs(stft(y=np.array(tmp), n_fft=N_FFT, hop_length=HOP_LEN))[np.newaxis,:,:]
        tmp = rolling_spectrogram(spec=tmp, winLen=args.winlen)
        for j in tmp:
            train_tmp.append( (j, label0) )
    train += train_tmp[:-len(tmp)]
    valid += train_tmp[-len(tmp):]
    train_tmp = []
    print('load data whose label 1...')
    for i in tqdm( d1 ):
        tmp = load(i, sr=SR)[0]
        tmp = np.abs(stft(y=np.array(tmp), n_fft=N_FFT, hop_length=HOP_LEN))[np.newaxis,:,:]
        tmp = rolling_spectrogram(spec=tmp, winLen=args.winlen)
        for j in tmp:
            train_tmp.append( (j, label1) )
    train += train_tmp[:-len(tmp)]
    valid += train_tmp[-len(tmp):]
    
    shuffle(train)
    shuffle(valid)
    
    # - Set data iteration - #
    train_iter = chainer.iterators.SerialIterator(dataset=train, batch_size=args.batchsize, shuffle=True, repeat=True)
    valid_iter = chainer.iterators.SerialIterator(dataset=valid, batch_size=args.batchsize, shuffle=True, repeat=False)    
    
    # - Set trainer - #
    print("setup trainer...")
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    updater = chainer.training.StandardUpdater(iterator=train_iter, optimizer=optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater=updater, stop_trigger=(args.epoch, "epoch"), out=args.result)
    
    trigger = (1, "epoch")
    snap_trigger = (10, "epoch")
    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu), trigger=trigger)
    trainer.extend(extensions.LogReport(trigger=trigger), trigger=trigger)
    trainer.extend(extensions.PrintReport(
        ["epoch", "iteration", "main/loss", "validation/main/loss", "main/acc", "validation/main/acc", "elapsed_time"]), trigger=trigger)
    trainer.extend(extensions.ProgressBar(update_interval=2))
    trainer.extend(extensions.PlotReport(
        ["main/loss", "validation/main/loss"], "epoch", file_name="loss.png", trigger=trigger), trigger=trigger)
    trainer.extend(extensions.PlotReport(
        ["main/acc", "validation/main/acc"], "epoch", file_name="acc.png", trigger=trigger), trigger=trigger)
    trainer.extend(extensions.snapshot(), trigger=snap_trigger)
    
    # - Run training - #
    print('='*30)
    print('Train data: {}'.format(len(train)))
    print('Test data : {}'.format(len(valid)))
    print('Epoch num : {}'.format(args.epoch))
    print('Batch size: {}'.format(args.batchsize))
    print('Window Len: {}'.format(args.winlen))
    print('='*30)
    trainer.run()
    
    print("saving model...")
    model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.result, "model.npz"), model)
    chainer.serializers.save_npz(os.path.join(args.result, "optimizer.npz"), optimizer)
    
if __name__ == "__main__":
    main()