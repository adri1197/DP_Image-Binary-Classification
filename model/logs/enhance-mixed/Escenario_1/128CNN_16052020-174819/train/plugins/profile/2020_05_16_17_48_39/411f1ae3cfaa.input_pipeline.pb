	�&l?Y�z@�&l?Y�z@!�&l?Y�z@	���b�?���b�?!���b�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�&l?Y�z@`����?A܁:���z@Y�
(��G�?*	
ףp=�a@2F
Iterator::Model-Z��լ�?!��fNWVK@)}?5^�I�?1��1��F@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatN�G���?!$���}�7@)�3�ތ��?1T��Ϣ�5@:Preprocessing2S
Iterator::Model::ParallelMap��X���?!���%�"@)��X���?1���%�"@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�v1�t��?!;���Y+@)t^c��ފ?1�c^��"@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�L��O�?!be����F@)y�ߢ���?1�p$Ww@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[1]::TensorSlice��� y?!_G�^@)��� y?1_G�^@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�5x_�e?!�o��=�?)�5x_�e?1�o��=�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	`����?`����?!`����?      ��!       "      ��!       *      ��!       2	܁:���z@܁:���z@!܁:���z@:      ��!       B      ��!       J	�
(��G�?�
(��G�?!�
(��G�?R      ��!       Z	�
(��G�?�
(��G�?!�
(��G�?JCPU_ONLY