# implement  self-attention convolution LSTM

# 代码实验效果 
train loss 0.013609007415620404   
valid loss 0.01899987788250049    

![训练曲线](./outputs/train_val_loss_curve_epoch_29.png)   
![效果](./outputs/029_00800.png)  


## Conv LSTM

![base](./model/pics/conv_lstm_cell.png)

## self-attention ConvLSTM momory module

![sa-conv_LSMT](./model/pics/self_attention_memory_module.png)

## self-attention ConvLSTM Arch

![sa-conv_LSMT](./model/pics/self_attention_conv_LSTM.png)

## 多层堆叠

![sa-conv_LSMT](./model/pics/Conv_lstm.png)


References:

- https://github.com/MinNamgung/sa_convlstm
- https://github.com/hyona-yu/SA-convlstm
- Lin, Zhihui,. [《Self-Attention ConvLSTM for Spatiotemporal Prediction》](https://ojs.aaai.org/index.php/AAAI/article/view/6819)
- https://github.com/jerrywn121/TianChi_AIEarth/tree/main/SAConvLSTM
