�	�wJ�&@�wJ�&@!�wJ�&@		�y@	�y@!	�y@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�wJ�&@E�e�?�?1~�
�R!@I�Ϲ��R�?YoB@��?*	�x�&1`k@2F
Iterator::Model"m�OT6�?!#zw�(I@)e�����?1$f*�0]E@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap
��t�?!��G&v�4@);S��.�?1lsZ�ƥ.@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�H��rڻ?!݅��H@)�)��s�?1�=9e�X-@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat`��Ù�?!F��q�.,@)=,Ԛ��?1�^㩝$@:Preprocessing2S
Iterator::Model::ParallelMapO ���?!��P�<^@)O ���?1��P�<^@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[1]::TensorSlice;�ʃ��?!��iTK^@);�ʃ��?1��iTK^@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorH�`���?!��:�C@)H�`���?1��:�C@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?7.7 % of the total step time sampled is spent on Kernel Launch.*moderate2A9.3 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	E�e�?�?E�e�?�?!E�e�?�?      ��!       "	~�
�R!@~�
�R!@!~�
�R!@*      ��!       2      ��!       :	�Ϲ��R�?�Ϲ��R�?!�Ϲ��R�?B      ��!       J	oB@��?oB@��?!oB@��?R      ��!       Z	oB@��?oB@��?!oB@��?JGPU�"m
Pgradient_tape/03062020-210734/conv2d_4/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownv������?!v������?"j
Mgradient_tape/03062020-210734/max_pooling2d_3/MaxPool/MaxPoolGrad:MaxPoolGradUnknownޗ�M�?!�����?"U
8gradient_tape/03062020-210734/conv2d_4/ReluGrad:ReluGradUnknown��,�o�?!t�����?"E
(03062020-210734/conv2d_4/BiasAdd:BiasAddUnknownq��[��?!����
�?"m
Pgradient_tape/03062020-210734/conv2d_5/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown���ʲ?!�ts�W��?"?
"03062020-210734/conv2d_4/Relu:ReluUnknown���Z���?!Figt[��?"k
Ngradient_tape/03062020-210734/conv2d_5/Conv2DBackpropInput:Conv2DBackpropInputUnknownnͤa�?!'@��!Z�?"-
IteratorGetNext/_1_Send1���V�?!8�C}���?"C
&03062020-210734/conv2d_5/Conv2D:Conv2DUnknown~G��	f�?!���U�?"C
&03062020-210734/conv2d_4/Conv2D:Conv2DUnknown���x���?!̓b��?2blackI�a�2��?Qs��9l�X@"�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate?7.7 % of the total step time sampled is spent on Kernel Launch.moderate"A9.3 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 