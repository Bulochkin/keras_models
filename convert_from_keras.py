import os
import subprocess

from keras import backend as keras_backend
from keras.models import load_model

with keras_backend.tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #saver = K.tf.train.Saver(write_version=K.tf.train.SaverDef.V1)
    
    keras_backend.set_learning_phase(1) #set learning phase
    
    keras_model_path = '/Users/pavelbulochkin/banuba/keras_models/ruslan/epoch_99_time_1497458133.29.hdf5'
    out_file_name = os.path.splitext(os.path.basename(keras_model_path))[0]
    
    model = load_model(keras_model_path)
    model.load_weights(keras_model_path)
    
    in_tensor = model.input.name.split(':')[0]
    out_tensor = model.output.name.split(':')[0]
    
    out_folder_path = os.path.join(os.getcwd() + '/inference/')
    if not os.path.exists(out_folder_path) :
        os.makedirs(out_folder_path)
    
    out_file_name_raw = out_file_name +'_raw' + '.pb'
    out_file_path_raw = os.path.join(out_folder_path, out_file_name_raw)

    out_file_name_freezed = out_file_name + '_freezed' + '.pb'
    out_file_path_freezed = os.path.join(out_folder_path, out_file_name_freezed)

    out_file_name_transformed = out_file_name + '_transformed' + '.pb'
    out_file_path_transformed = os.path.join(out_folder_path, out_file_name_transformed)

    out_file_name_for_inference =  out_file_name + '_for_inference' + '.pb'
    out_file_path_for_inference = os.path.join(out_folder_path, out_file_name_for_inference)

    ckpt_out_raw_file_path = os.path.join(out_folder_path, out_file_name_raw + '.ckpt')
    
    saver = keras_backend.tf.train.Saver()
    saver.save(sess, ckpt_out_raw_file_path)
    
    keras_backend.tf.train.write_graph(sess.graph.as_graph_def(), logdir=out_folder_path, name=out_file_name_raw)


tools_path = '/Users/pavelbulochkin/banuba/keras_models/tensorflow/bazel-bin/tensorflow/python/tools/'
tools_path2 = '/Users/pavelbulochkin/banuba/keras_models/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph'
#--input_binary=True

freeze_string = """{}/freeze_graph --input_graph={} --input_checkpoint={} --output_graph={} \
    --output_node_names='{}' """.format(tools_path, out_file_path_raw, ckpt_out_raw_file_path, out_file_path_freezed, out_tensor)

#transform_string = """{} --in_graph={} --out_graph={} --inputs={} --outputs={} \
#    --transforms='strip_unused_nodes(type=float) remove_nodes(op=Identity, op=CheckNumerics)
#    fold_constants(ignore_errors=true)'""".format(tools_path2, out_file_path_freezed, out_file_path_transformed, in_tensor, out_tensor)

inference_string = """{}/optimize_for_inference --input={} --output={}  --input_names='{}' \
    --output_names='{}'""".format(tools_path, out_file_path_freezed, out_file_path_for_inference, in_tensor, out_tensor)

print('Executing :', freeze_string)
os.system(freeze_string)
print('-'*30)

#print('Executing :', transform_string)
#os.system(transform_string)
#print('-'*30)

print('Executing :', inference_string)
os.system(inference_string)
print('-'*30)

print("In Tensor:", in_tensor)
print("Out Tensor:", out_tensor)
