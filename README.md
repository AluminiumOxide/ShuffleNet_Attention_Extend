# ShuffleNet_Attention_Extend
We add attention mechanism come from Squeeze-and-Excitation and Selective Kernel to InvertedResidual block in shuffleNetV2.

which can be found in the dir about 'shuffle_SE' and 'shuffle_SK'

the main directory build as
> proto_1_vggnet
> 
> proto_2_googlenet
> 
> proto_3_resnet
> 
> proto_4_densenet
> 
> proto_5_mobilenet
> 
> remix_1_shufflenet
> 
> shuffle_SE
> 
> shuffle_SK
> 
> > model.py
> >
> > predict.py
> >
> > train.py

The difference between these Net comes in 'model.py'

Additionally, we backup the train info in the '/draw_plot', Hope these a bit help you understand the code.

Cervical cancer screening dataset used for classification located in '\data\cancer_data', which split into train and val.

We fetch 10 example cervical cancer image in each categories(Although these pictures not enough to support training).

If you need a complete cervical cancer dataset or make a deeper academic exchanges, please send an email to AluminiumOxide@163.com.

