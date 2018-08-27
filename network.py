# ===== modules ===== #
# chainer
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import reporter
import numpy as np

# ===== setup ===== #
cp = chainer.cuda.cupy

# ===== Spec classification model ===== #
class Spec_classification(Chain):
    def __init__(self):
        super(Spec_classification, self).__init__()
        with self.init_scope():
            # Convolution layers
            self.conv1 = L.Convolution2D(in_channels=None, out_channels=16, ksize=(3,4), stride=(1,2), pad=(1,1))
            self.conv2 = L.Convolution2D(in_channels=None, out_channels=32, ksize=(3,4), stride=(1,2), pad=(1,1))
            self.conv3 = L.Convolution2D(in_channels=None, out_channels=64, ksize=(3,4), stride=(1,2), pad=(1,1))
            self.conv4 = L.Convolution2D(in_channels=None, out_channels=128, ksize=(3,4), stride=(1,2), pad=(1,1))
            #self.conv5 = L.Convolution2D(in_channels=None, out_channels=256, ksize=(2,4), stride=(1,2), pad=(1,1))
            #self.conv6 = L.Convolution2D(in_channels=None, out_channels=512, ksize=4, stride=2, pad=1)

            # Batch normalization layers
            self.bn1 = L.BatchNormalization(16)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(64)
            self.bn4 = L.BatchNormalization(128)
            #self.bn5 = L.BatchNormalization(256)
            #self.bn6 = L.BatchNormalization(512)

            # Fully connected layers
            self.lc1 = L.Linear(in_size=None, out_size=1024)
            self.lc2 = L.Linear(in_size=None, out_size=128)
            self.lc3 = L.Linear(in_size=None, out_size=2)

    def __call__(self, X, y):
        # Extract features
        yhat = self.extract_feature(X)
        
        # Compute eval functions
        loss = F.softmax_cross_entropy(yhat, y)
        acc = F.accuracy(yhat, y)

        reporter.report({'loss': loss, 'acc': acc}, self)
        return loss

    def extract_feature(self, X):
        # Convolution layers
        h = F.relu( self.bn1(self.conv1(X)) )
        h = F.relu( self.bn2(self.conv2(h)) )
        h = F.relu( self.bn3(self.conv3(h)) )
        h = F.relu( self.bn4(self.conv4(h)) )
        #h = F.relu( self.bn5(self.conv5(h)) )
        #h = F.relu( self.bn6(self.conv6(h)) )
        h = F.dropout(F.relu( self.lc1(h) ), ratio=0.5)
        h = F.dropout(F.relu( self.lc2(h) ), ratio=0.5)
        h = self.lc3(h)
        return h
