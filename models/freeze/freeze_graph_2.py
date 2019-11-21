import tensorflow as tf
import argparse
import uuid
import tensorflow.compat.v1.keras.backend as K
import tensorflow.keras as keras
import shutil
import os
import subprocess
from pathlib import Path

anaconda_path = Path("C:/Program Files (x86)/Microsoft Visual Studio/Shared/Anaconda3_64/")
user_path = Path("C:/users/adrián/appdata/local/programs/python/python37/")
tf_tools_path = os.path.join(user_path, "Lib/site-packages/tensorflow/python/tools/")
freeze_script = os.path.join(tf_tools_path, "freeze_graph.py")

parser = argparse.ArgumentParser(description="Convert Keras .h5 file to Tensorflow .pb frozen model.")
parser.add_argument("--h5", type=str, default="model.h5", help="Keras .h5 file (input)")
parser.add_argument("--pb", type=str, default="model.pb", help="Tensorflow frozen model (output)")
args = parser.parse_args()

K.set_learning_phase(0)

model = keras.models.load_model(args.h5)
model_output = model.output.op.name
tempdir = str(uuid.uuid4())
os.mkdir(tempdir)
try:
    temp_tf_file = str(os.path.join(tempdir, "tf.ckpt"))
    saver = tf.compat.v1.train.Saver()
    saver.save(K.get_session(), temp_tf_file)
    tmp_meta_file = str(temp_tf_file) + ".meta"

    saver = tf.compat.v1.train.import_meta_graph(tmp_meta_file, clear_devices=True)
    K.get_session().run(tf.compat.v1.global_variables_initializer())
    K.get_session().run(tf.compat.v1.local_variables_initializer())
    sess = K.get_session()
    saver.restore(sess, tf.train.latest_checkpoint(os.path.expanduser(tempdir)))

    output_node_names = model_output

    # for fixing the bug of batch norm
    gd = sess.graph.as_graph_def()
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, gd, output_node_names.split(","))
    tf.io.write_graph(converted_graph_def, ".", args.pb, as_text=False)
    print("Done.")
finally:
    shutil.rmtree(tempdir)