import tensorflow as tf
import numpy as np

num_samples = 1000 #Number of classes
lr = 0.0001 #Learning rate
num_frame = 250
num_frame_inteval = 20
steps = 5100 #optimization steps

z1 = tf.get_variable("class_1_v", [num_samples, 2], initializer=tf.contrib.layers.xavier_initializer()) #dxL
z2 = tf.get_variable("class_2_v", [num_samples, 2], initializer=tf.contrib.layers.xavier_initializer()) #dxL
z3 = tf.get_variable("class_3_v", [num_samples, 2], initializer=tf.contrib.layers.xavier_initializer()) #dxL

# z1 = (z1 / tf.norm(tf.reduce_mean(z1, axis=-1)))*tf.minimum(1.,tf.norm(tf.reduce_mean(z1, axis=-1)))
# z2 = (z2 / tf.norm(tf.reduce_mean(z2, axis=-1)))*tf.minimum(1.,tf.norm(tf.reduce_mean(z1, axis=-1)))
# z3 = (z3 / tf.norm(tf.reduce_mean(z3, axis=-1)))*tf.minimum(1.,tf.norm(tf.reduce_mean(z1, axis=-1)))

# z1 = (z1 / tf.norm(z1, axis=1, keepdims=True)) * tf.minimum(1.,tf.norm(z1, axis=1, keepdims=True))
# z2 = (z2 / tf.norm(z2, axis=1, keepdims=True)) * tf.minimum(1.,tf.norm(z2, axis=1, keepdims=True))
# z3 = (z3 / tf.norm(z3, axis=1, keepdims=True)) * tf.minimum(1.,tf.norm(z3, axis=1, keepdims=True))

z1 = (z1 / tf.norm(z1, axis=1, ord=2, keepdims=True))
z2 = (z2 / tf.norm(z2, axis=1, ord=2, keepdims=True)) 
z3 = (z3 / tf.norm(z3, axis=1, ord=2, keepdims=True))

# z1z1 = tf.matmul(z1, z1, transpose_a=True)
# z2z2 = tf.matmul(z2, z2, transpose_a=True)
# z3z3 = tf.matmul(z3, z3, transpose_a=True)
# z1z2 = tf.matmul(z1, z2, transpose_a=True)
# z1z3 = tf.matmul(z1, z3, transpose_a=True)
# z2z3 = tf.matmul(z2, z3, transpose_a=True)

# diag11 = tf.diag_part(z1z1)
# diag12 = tf.diag_part(z2z2)
# diag13 = tf.diag_part(z3z3)


mean1 = tf.reduce_mean(z1, axis=0)
mean2 = tf.reduce_mean(z2, axis=0)
mean3 = tf.reduce_mean(z3, axis=0)

mean_dis12 = tf.norm(mean1-mean2) 
mean_dis13 = tf.norm(mean1-mean3) 
mean_dis23 = tf.norm(mean2-mean3) 

# cost = - tf.reduce_sum((z1z1 + z2z2 + z3z3)) + 2 * tf.reduce_sum((z1z2 + z1z3 + z2z3)) - num_samples*tf.reduce_sum(diag11+diag12+diag13)
# cost = cost / num_samples

delta_1 = tf.tile(tf.expand_dims(z1,1),[1,num_samples,1]) - tf.tile(tf.expand_dims(z1,0),[num_samples,1,1])                                                                                                                            
distance_1 = tf.reduce_sum(delta_1*delta_1)
delta_2 = tf.tile(tf.expand_dims(z2,1),[1,num_samples,1]) - tf.tile(tf.expand_dims(z2,0),[num_samples,1,1])                                                                                                                            
distance_2 = tf.reduce_sum(delta_2*delta_2)
delta_3 = tf.tile(tf.expand_dims(z3,1),[1,num_samples,1]) - tf.tile(tf.expand_dims(z3,0),[num_samples,1,1])                                                                                                                            
distance_3 = tf.reduce_sum(delta_3*delta_3)

delta_12 = tf.tile(tf.expand_dims(z1,1),[1,num_samples,1]) - tf.tile(tf.expand_dims(z2,0),[num_samples,1,1])                                                                                                                            
distance_12 = tf.reduce_sum(delta_12*delta_12)
delta_13 = tf.tile(tf.expand_dims(z1,1),[1,num_samples,1]) - tf.tile(tf.expand_dims(z3,0),[num_samples,1,1])                                                                                                                            
distance_13 = tf.reduce_sum(delta_13*delta_13)
delta_23 = tf.tile(tf.expand_dims(z2,1),[1,num_samples,1]) - tf.tile(tf.expand_dims(z3,0),[num_samples,1,1])                                                                                                                            
distance_23 = tf.reduce_sum(delta_23*delta_23)

cost = (num_samples*1.0/(num_samples-1))*(distance_1 + distance_2 + distance_3) - (distance_12 + distance_13 + distance_23)


opt = tf.train.AdamOptimizer(learning_rate=lr)
opt_op = opt.minimize(cost)

savez1 = np.zeros((num_frame,num_samples,2))
savez2 = np.zeros((num_frame,num_samples,2))
savez3 = np.zeros((num_frame,num_samples,2))

count = 0
with tf.Session() as sess:
	sess.run(tf.initializers.global_variables())
	for i in range(steps):
		if i%num_frame_inteval==0 and count<num_frame:
			z1out,z2out,z3out = sess.run([z1,z2,z3])
			savez1[count] = z1out
			savez2[count] = z2out
			savez3[count] = z3out
			count += 1
		_, loss, md12, md13, md23 = sess.run([opt_op, cost, mean_dis12, mean_dis13, mean_dis23])
		print('Step %d, loss: %f, md12: %f, md23: %f, md13: %f'%(i, loss, md12, md13, md23))

	
import scipy.io as sio
sio.savemat('/mfs/tianyu/google/tianyu/siamese_to_MMLDA/class1.mat', {'z1': savez1})
sio.savemat('/mfs/tianyu/google/tianyu/siamese_to_MMLDA/class2.mat', {'z2': savez2})
sio.savemat('/mfs/tianyu/google/tianyu/siamese_to_MMLDA/class3.mat', {'z3': savez3})

#CUDA_VISIBLE_DEVICES=0 python craft_M3LDA_means.py	