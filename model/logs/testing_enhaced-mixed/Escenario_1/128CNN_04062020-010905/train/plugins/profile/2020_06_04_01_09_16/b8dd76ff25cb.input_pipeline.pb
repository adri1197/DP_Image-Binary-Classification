	�n��c&@�n��c&@!�n��c&@	���oY@���oY@!���oY@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�n��c&@�W���?1����Qb!@ILnYk��?Y���N��?*	j�t��_@2F
Iterator::ModelI�2�單?!:$g�G@)#��<�?1�����A@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapIJzZ��?!�$�v�?@)�U�3��?1��Y��Z;@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�,AF@��?!Ԡ&�s.@)���B�?1�� ׈�*@:Preprocessing2S
Iterator::Model::ParallelMap���ek�?!�0h��&@)���ek�?1�0h��&@:Preprocessing2X
!Iterator::Model::ParallelMap::ZipOv3��?!��ۘiJ@)�Za�~?1"5f��@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::TensorSlice��)1	w?!��)���@)��)1	w?1��)���@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��[;Qb?!�A/�X��?)��[;Qb?1�A/�X��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?9.9 % of the total step time sampled is spent on Kernel Launch.*moderate2A6.7 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�W���?�W���?!�W���?      ��!       "	����Qb!@����Qb!@!����Qb!@*      ��!       2      ��!       :	LnYk��?LnYk��?!LnYk��?B      ��!       J	���N��?���N��?!���N��?R      ��!       Z	���N��?���N��?!���N��?JGPU