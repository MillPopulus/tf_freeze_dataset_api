# README

This function can auto search the hard-related dataset nodes in the graph which not searched in the native function of tensorflow, such as "MakeIterator" node, "IteratorFromStringHandle" node, "RangeDataset" node and so on.

Hard-related: connected in the graph, not by the placeholder.

## usage
```
from freeze_compatible_with_dataset_api import convert_variables_to_constants

convert_variables_to_constants(..., output_node_names=["xxx"], dataset_flag=True)
```

Please construct dataset nodes in a seperate variable scope, in order to get the important nodes conveniently after loading the .pb file. 

### example 1

#### save part
```
  with tf.variable_scope('input'):
      data=tf.placeholder(tf.int32, shape=[None,MAX_LEN], name='data')
      label=tf.placeholder(tf.int32, shape=[None,1], name='label')
      batch_size=tf.placeholder(tf.int64, name='batch_size')
      iterator=tf.data.Dataset.from_tensor_slices((data, label)).batch(batch_size).make_initializable_iterator()
  return iterator.initializer, iterator.get_next(), data, label, batch_size
  #......
  with tf.Session() as sess:
      #......
      output_graph_def= convert_variables_to_constants(sess, sess.graph_def, output_node_names=["cnn/pred"], dataset_flag=True)
      tf.train.write_graph(output_graph_def, './', 'graph_bin.pb', as_text=False)
```

#### load part
```
graph_def = tf.GraphDef()
graph_def.ParseFromString(open("./graph_bin.pb", "rb").read())
self.batch_size_ph=tf.placeholder(tf.int64)
self.pred = tf.import_graph_def(graph_def, input_map={'cnn/keep_prob:0': 1.0, 'input/batch_size:0': self.batch_size_ph}, return_elements=['cnn/pred:0'], name='imported_predictor')[0]
self.data_ph = tf.get_default_graph().get_tensor_by_name("imported_predictor/input/data:0")
self.label_ph = tf.get_default_graph().get_tensor_by_name("imported_predictor/input/label:0")
self.iter_initializer= tf.get_default_graph().get_operation_by_name("imported_predictor/input/MakeIterator") 
```

### example 2
When you use feedable iterator, because the training_iterator, validation_iterator and the handle hard-related with the main graph are connected by placeholder, you need to name the training string handle and the validation string handle and add them to the output_node_names arg manually.

#### save part
```
train_data = tf.data.Dataset.range(6)
val_data = tf.data.Dataset.range(5)

handle = tf.placeholder(tf.string,shape=[], name="iterator_handle")

iterator = tf.data.Iterator.from_string_handle(
    train_data.output_types,train_data.output_shapes)

next_element = iterator.get_next()

train_iterator_handle = train_data.make_one_shot_iterator().string_handle(name='train_sh')
val_iterator_handle = val_data.make_initializable_iterator().string_handle(name='val_sh')
#......

with tf.Session() as sess:
    #......
    output_graph_def= convert_variables_to_constants(sess, sess.graph_def, output_node_names=['cnn/pred','train_sh','val_sh'], dataset_flag=True)
    tf.train.write_graph(output_graph_def, './', 'graph_bin.pb', as_text=False)
```

use tf.get_default_graph().get_tensor_by_name to load the handle placeholder, and the two string_handles.
