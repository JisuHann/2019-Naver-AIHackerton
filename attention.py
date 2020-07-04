import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

	#mask : 현재 적용 포지션 이후에 있는 포지션에 attention 설정 못하게 만드는 기법
    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # attn 벡터 생성하기 : (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len) > attention 값 계산(bmm 은 행렬값 용이하게 만듦)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        #활성함수 적용
        attn = F.softmax(attn.view(-1, input_size),dim=1).view(batch_size, -1, input_size)

        # attn 벡터, context 적용된 output 생성 (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim) > context vector 출력
        mix = torch.bmm(attn, context)
        # concat -> (batch, out_len, 2*dim) (cat : 텐서 결합)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim) : 활성함수 실행
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
