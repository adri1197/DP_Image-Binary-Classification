	�,	PS{%@�,	PS{%@!�,	PS{%@	n��Ke# @n��Ke# @!n��Ke# @"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�,	PS{%@�Y�X�?1X歺� @I�:���?Y@j'���?*	D�l���@2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapu<f�2��?!'�@�GN@)�]�pX�?17-�fG@:Preprocessing2s
<Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map�	MK�?!��%��/:@)d��A%�?1c�W���5@:Preprocessing2F
Iterator::Model^�/�ۮ?!�;9�&@)/m8,��?1����
�!@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�G�C��?!r��y[@)Ts��P��?1�h���@:Preprocessing2n
7Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetchn�|�b��?!�l�Z @)n�|�b��?1�l�Z @:Preprocessing2�
JIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatk`��Ù?!*8��p@)��I��Ж?1X8� �T@:Preprocessing2S
Iterator::Model::ParallelMap��ݯ|�?!�,B)� @)��ݯ|�?1�,B)� @:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatf1���6�?!w�G���?)�z��9y�?12b����?:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�l#���?!�?�bO@)���n-s?1sU�l�s�?:Preprocessing2�
QIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeQ�[��g?!��n����?)Q�[��g?1��n����?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice������`?!�U V�?)������`?1�U V�?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[1]::ConcatenateO;�5Y�n?!�Î���?) �t���[?1�
���?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�}���U?!&R��a�?)�}���U?1&R��a�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor�*5{�H?!�*a=�?)�*5{�H?1�*a=�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?9.8 % of the total step time sampled is spent on Kernel Launch.*moderate2A3.7 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Y�X�?�Y�X�?!�Y�X�?      ��!       "	X歺� @X歺� @!X歺� @*      ��!       2      ��!       :	�:���?�:���?!�:���?B      ��!       J	@j'���?@j'���?!@j'���?R      ��!       Z	@j'���?@j'���?!@j'���?JGPU