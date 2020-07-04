import random
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

#장치 정상 확인 여부
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

	#decoder 부분 생성자 
    def __init__(self, vocab_size, max_len, hidden_size,sos_id, eos_id,n_layers=1, rnn_cell='gru', bidirectional=False,input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.bidirectional_encoder = bidirectional

		#2. gru 정의
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
				
		#3. Embedding 정의
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
		#4. Attention.py 정의
        if use_attention:
            self.attention = Attention(self.hidden_size)
				
		#5. Linear 정의 - output 출력
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
		
        #1. Embedding
        embedded = self.embedding(input_var)
			
        #2. input_dropout
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()
		
        #3. gru
        output, hidden = self.rnn(embedded, hidden)
				
		#4. attention 실행 부분 :decoder의 output과 encoder_outputs(context) 적용한 결과물 output과 attn 생성
        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)
		
		#Linear와 softmax 처리
        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        
        return predicted_softmax, hidden, attn

	#전체 forward 실행
    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, teacher_forcing_ratio=0):
        
        #일단은 코드 정렬

        #ret_dict : attention 내용 저장한 리스트
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

		#파라미터 정리
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio)
		
        decoder_hidden = self._init_state(encoder_hidden)
				
		#teacher_forcing 여부 체크
        use_teacher_forcing = True 
        if random.random() < teacher_forcing_ratio else False   #random.random()(0부터 1사이의 랜덤 값)으로 왜 했지...?

        decoder_outputs = []
        sequence_symbols = []  # sequence_sybols는 최종 결과가 담긴리스트(에 symbol 저장)
        lengths = np.array([max_length] * batch_size)
				
		#나온 decoder_output에 따라 symbol 판정
        def decode(step, step_output, step_attn):
            #step_output을 decoder_outputs에 저장
            decoder_outputs.append(step_output)
            if self.use_attention:
                #step_attn 결과를 retdict에 저장
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            #sequence_sybols는 최종 결과가 담긴리스트(에 symbol 저장)
            symbols = decoder_outputs[-1].topk(1)[1] #longtensor로 반환
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id) #끝에 있는 건지 판단해줌.
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        #실제 실행 부분
        
        if use_teacher_forcing:  # Teacher_forcing 실행시 predict된 값 + decoder_input이용
            #forward_step 실행 : forward_step 결과 decoder_output = predicted_softmax, decoder_hidden = hidden, attn = attn
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function)
            for di in range(decoder_output.size(1)):
                #step_output <= decoder_output(decode에 나오는 decoder_outputs과는 다름)
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :] #step에 해당하는 attention 값
                else:
                    step_attn = None
                decode(di, step_output, step_attn)

        else: #teacher_forcing이 없으므로 predict된 값만을 사용 : 이해가 잘 되지는 않지만 그냥 tensor 관련 조정 내용
            decoder_input = inputs[:, 0].unsqueeze(1)#start를 맞추기 위해서 
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                #다음 값에 들어갈 수 있도록 decoder_input 사용
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

	#hidden state 가져오는 방식
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

	#_init_state(hidden state 초기화시)에서 이용
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

	#코드 초기 실행시 이용 : 값들 정리하는 용도 : self._validate_args(inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio)
    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size, batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length, input, teacher_forcing_ratio 정리
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
