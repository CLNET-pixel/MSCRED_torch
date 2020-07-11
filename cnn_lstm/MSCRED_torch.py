import torch as t
from torch import nn
from torch.nn import functional as F
from cnn_lstm.convlstm import ConvLSTM
from cnn_lstm import utils


class CNNEncoder(nn.Module):
    """
    TO DO
    :param x (step_max,30,30,3)
    :return out
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=(1,1))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1,1))
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=(1,1))
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=(0,0))

    def forward(self, x):
        cnn1_out = F.selu(self.conv_1(x))
        cnn2_out = F.selu(self.conv_2(cnn1_out))
        cnn3_out = F.selu(self.conv_3(cnn2_out))
        cnn4_out = F.selu(self.conv_4(cnn3_out))
        return cnn1_out.unsqueeze(0), cnn2_out.unsqueeze(0), cnn3_out.unsqueeze(0), cnn4_out.unsqueeze(0)


class ConvLSTMAttentionLayer(nn.Module):
    """
    input:(1, 5, 32, 30, 30)
    output:
    input_dim=32,hidden_dim=32, kernel_size=(3,3), num_layers=1, batch_first=True, bias=True
    input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True,bias=True
    input_dim=128, hidden_dim=128, kernel_size=(3, 3), num_layers=1, batch_first=True,bias=True
    input_dim=256, hidden_dim=256, kernel_size=(3, 3), num_layers=1, batch_first=True,bias=True
    """
    def __init__(self, *args):
        super(ConvLSTMAttentionLayer, self).__init__()
        self.conv_lstm = ConvLSTM(args[0],args[1],args[2],args[3],args[4],args[5])

    def forward(self, input_data):
        outputs = self.conv_lstm(input_data)[0][0] # outputs is (1, 5, 32, 30, 30)
        # attention based on inner-product between feature representation of last step and other steps
        attention_w = []
        for k in range(utils.step_max):
            attention_w.append(t.sum(t.mul(outputs[0][k], outputs[0][-1])) / utils.step_max)  #只能支持batch-size=1
        attention_w = F.softmax(t.stack(attention_w), dim=0).view(1, utils.step_max)

        outputs = outputs[0].view(utils.step_max, -1)
        outputs = t.matmul(attention_w, outputs)
        outputs = outputs.view(1, input_data.shape[2], input_data.shape[3], input_data.shape[4])

        return outputs


class ConvLSTMAttention(nn.Module):
    def __init__(self):
        super(ConvLSTMAttention, self).__init__()
        self.convlstm_1 = ConvLSTMAttentionLayer(32, 32, (3,3), 1, True, True)
        self.convlstm_2 = ConvLSTMAttentionLayer(64, 64, (3,3), 1, True, True)
        self.convlstm_3 = ConvLSTMAttentionLayer(128, 128, (3,3), 1, True, True)
        self.convlstm_4 = ConvLSTMAttentionLayer(256, 256, (3, 3), 1, True, True)

    def forward(self, *input_data):
        out_1 = self.convlstm_1(input_data[0])
        out_2 = self.convlstm_2(input_data[1])
        out_3 = self.convlstm_3(input_data[2])
        out_4 = self.convlstm_4(input_data[3])

        return out_1, out_2, out_3, out_4


class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.deconv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=(2,2), stride=2)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=64,kernel_size=(3,3), stride=2, padding=(1,1))
        self.deconv_2 = nn.ConvTranspose2d(in_channels=128, out_channels=32,kernel_size=(2,2), stride=2)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=64, out_channels=3,kernel_size=(3,3), stride=1,padding=(1,1))

    def forward(self, *input_data):
        out_4 = F.selu(self.deconv_4(input_data[3]))
        out_3 = F.selu(self.deconv_3(t.cat((input_data[2], out_4), dim=1)))
        out_2 = F.selu(self.deconv_2(t.cat((input_data[1], out_3), dim=1)))
        out_1 = self.deconv_1(t.cat((input_data[0], out_2), dim=1))
        return out_1


class MSCRED(nn.Module):
    def __init__(self):
        super(MSCRED, self).__init__()
        self.encoder = CNNEncoder()
        self.attention = ConvLSTMAttention()
        self.decoder = CNNDecoder()

    def forward(self, input_data):
        cnn1_out, cnn2_out, cnn3_out, cnn4_out = self.encoder(input_data)
        out_1, out_2, out_3, out_4 = self.attention(cnn1_out, cnn2_out, cnn3_out, cnn4_out)
        out = self.decoder(out_1, out_2, out_3, out_4)
        return out


if __name__=='__main__':
    pass
    # test nn.Conv2d
    conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=1, bias=False, padding=(0, 0))
    x = t.ones(1, 3, 30, 30)
    output = conv_1(x)
    # test the class
    cnn = CNNEncoder()
    # x = t.ones(5,3,30,30)
    # output = cnn(x)
    # print(output[0].size(),output[1].size(),output[2].size(),output[3].size())

    # test attention
    # x = t.rand(1, 5, 256, 4, 4)
    # attention = ConvLSTMAttentionLayer(256, 256, (3, 3), 1, True, True)
    # print(attention(x).size())

    # cnndecoder = nn.ConvTranspose2d(in_channels=64, out_channels=3,kernel_size=(3,3), stride=1,padding=(1,1))
    # x = t.rand(1,64,30,30)
    # output = cnndecoder(x)
    # print(output.size())

    # test MSCRED
    # model = MSCRED()
    # x = t.rand(5, 3, 30, 30)
    # output = model(x)
    # print(output.size())

