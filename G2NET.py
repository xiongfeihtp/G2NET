import tensorflow as tf
from tensorflow.contrib import layers


class SelfAttention:
    """SelfAttention class"""

    def __init__(self,
                 dim,
                 key_mask,
                 query_mask,
                 length,
                 M=None):

        self.key_mask = key_mask
        self.query_mask = query_mask
        self.length = length
        self.dim = dim
        self.M = M

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            output = self.scaled_dot_product(inputs, inputs, inputs)
            return output

    def scaled_dot_product(self, qs, ks, vs):

        o1 = tf.matmul(qs, ks, transpose_b=True)

        if self.M is not None:
            M = tf.expand_dims(self.M, axis=1)
            o1 = o1 + M
        o2 = o1 / (self.dim ** 0.5)

        if self.key_mask is not None:
            padding_num = -2 ** 32 + 1
            mask = tf.expand_dims(self.key_mask, 1)
            mask = tf.tile(mask, [1, self.length, 1])
            paddings = tf.ones_like(o2) * padding_num
            o2 = tf.where(tf.equal(mask, 0), paddings, o2)

        o3 = tf.nn.softmax(o2)

        if self.query_mask is not None:
            mask = tf.expand_dims(self.query_mask, 2)
            o3 = o3 * tf.cast(mask, tf.float32)

        return tf.matmul(o3, vs)


class GRE:
    """Geography Interaction Encoder"""

    def __init__(self):
        pass

    def build(self, nodes_repr, edges_repr, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            if edges_repr is not None:
                M = tf.reduce_mean(edges_repr, axis=3)
            else:
                M = None

            trans_block = SelfAttention(
                key_mask=None,
                query_mask=None,
                length=4,
                dim=32,
                M=M
            )

            gsi_repr = tf.layers.flatten(trans_block.build(nodes_repr, reuse=False, scope='gre_trans'))

            input = layers.fully_connected(gsi_repr, 64, activation_fn=tf.nn.relu, scope='gre_ffn_1')

            input = layers.fully_connected(input, 32, activation_fn=tf.nn.relu, scope='gre_ffn_2')

            gsi_weight = layers.fully_connected(input, 1, activation_fn=None, scope='gre_ffn_3')

            return gsi_repr, gsi_weight


class G2NET:
    def __init__(self):
        pass
