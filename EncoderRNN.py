import sys
import math
import torch.nn as nn
from .baseRNN import BaseRNN

class EncoderRNN(BaseRNN):

	#EncoderRNN 생성자 설정
    def __init__(self, feature_size, hidden_size, input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(EncoderRNN, self).__init__(0, 0, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
 
        self.conv = nn.Sequential(
			#구축해야 할 레이어 넣는 부분: resnet18이나 densenet등의 가벼운 layer 추가(CNN 종류 찾기)
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32 , kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)), 
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

		#특성 개수 정의
        feature_size = math.ceil((feature_size - 11 + 1 + (5*2)) / 2) #conv2d 거치므로 2번
        feature_size = math.ceil(feature_size - 11 + 1 + (5*2))
        feature_size *= 32

		#gru 실행 부분
        self.rnn = self.rnn_cell(feature_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

	#Encoder 진행 부분 forward
    def forward(self, input_var, input_lengths=None):  # data loader(B, time,128)
        input_var = input_var.unsqueeze(1) #(B=64,C=32,Time, depth)
        
		#Sequential 실행 부분
        x = self.conv(input_var)

        # BxCxTxD => B(atch)xT(ime)xC(conv)xD(depths) : 다시보기
        x = x.transpose(1, 2)
        x = x.contiguous()
        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[2] * sizes[3])

        if self.training:
            self.rnn.flatten_parameters()
		
        #gru 실행
        output, hidden = self.rnn(x)

        return output, hidden
