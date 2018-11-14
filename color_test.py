#color3.py 테스트 모델

import tensorflow as tf
import numpy as np
import time
import sys


#DATA_NUM = 175104
"""실험 데이터 불러오기"""
with open("./data/data_all.txt",'r') as input_data:
    input_data_0 = input_data.readlines()


#text 문자열 데이터를 정수형으로 바꿔줌
for i in range(len(input_data_0)):
    input_data_0[i] = input_data_0[i][5:-2]
    input_data_0[i] = input_data_0[i].split(",")
    input_data_0[i] = [int(a) for a in input_data_0[i]]

x_data = []
y_data = []

for i in range(len(input_data_0)):           #x : B,C를 합친 데이터(input data), y : P 데이터(정답 데이터)
    if i%3==0:          #P_data
        y_data.extend(input_data_0[i])
    elif i%3==1:        #B_data
        x_data.extend(input_data_0[i])
    else:               #C_data
        x_data.extend(input_data_0[i])



print("y data : ", len(y_data))
#Train, Test, Validation 데이터 나누기 85%, 10%, 5%로 나눔
DATA_NUM = int(len(y_data)/3)
DATA_TRAIN_NUM = int(DATA_NUM)
DATA_TEST_NUM = int(DATA_NUM*0.1)
DATA_VAL_NUM = int(DATA_NUM*0.05)
print("DATA NUM : ",DATA_NUM)
print("DATA_TRAIN_NUM : ",DATA_TRAIN_NUM)
print("DATA_TEST_NUM : ",DATA_TEST_NUM)
print("DATA_VAL_NUM : ",DATA_VAL_NUM)
print("DATA_NUM_SUM : ",DATA_TEST_NUM+DATA_TRAIN_NUM+DATA_VAL_NUM)
"""
x_train_data = x_data[:(DATA_TRAIN_NUM*3*2)]
x_test_data = x_data[(DATA_TRAIN_NUM*3*2):-(DATA_VAL_NUM*3*2)]
x_val_data = x_data[(-DATA_VAL_NUM*3*2):]

y_train_data = y_data[:(DATA_TRAIN_NUM*3)]
y_test_data = y_data[(DATA_TRAIN_NUM*3):-(DATA_VAL_NUM*3)]
y_val_data = y_data[-DATA_VAL_NUM*3:]
"""
x_train_data_ = np.array(x_data[:(DATA_TRAIN_NUM*3*2)], dtype='float32')

y_train_data_ = np.array(y_data[:(DATA_TRAIN_NUM*3)], dtype='float32')

#데이터 shape 변경
x_train_data = np.reshape(x_train_data_, (-1,6))
y_train_data = np.reshape(y_train_data_, (-1,3))
print(x_train_data[:10])


#B
x_train_data[:,0] = 98      #B
x_train_data[:,1] = 90     #G
x_train_data[:,2] = 100     #R

#P
x_train_data[0:,3] = 55      #B
x_train_data[0:,4] = 23     #G
x_train_data[0:,5] = 33     #R
x_train_data[1:,3] = 207      #B
x_train_data[1:,4] = 95    #G
x_train_data[1:,5] = 255     #R
x_train_data[2:,3] = 175     #B
x_train_data[2:,4] = 111    #G
x_train_data[2:,5] = 79     #R
x_train_data[3:,3] = 95     #B
x_train_data[3:,4] = 111    #G
x_train_data[3:,5] = 15     #R


DOWNSIZE = 255

x_train_data = np.divide(x_train_data,DOWNSIZE)
y_train_data = np.divide(y_train_data,DOWNSIZE)


print(np.shape(y_train_data))

"""모델 설계"""


#일단 2 layer로 1층은 6X30, 2층은 30X3으로 제작 사이사이 relu 함수를 넣어줌
x = tf.placeholder(tf.float32, [None,6])


W_1 = tf.get_variable(shape=[6,50], name= 'W_1')
b_1 = tf.get_variable(shape=[50], name= 'b_1')

W_2 = tf.get_variable(shape=[50,100], name= 'W_2')
b_2 = tf.get_variable(shape=[100], name= 'b_2')

W_3 = tf.get_variable(shape=[100,150], name= 'W_3')
b_3 = tf.get_variable(shape=[150], name= 'b_3')

W_4 = tf.get_variable(shape=[150,3], name= 'W_4')
b_4 = tf.get_variable(shape=[3], name= 'b_4')


y_1 = tf.matmul(x,W_1) + b_1
y_r1 = tf.nn.sigmoid(y_1)

y_2 = tf.matmul(y_r1, W_2) + b_2
y_r2 = tf.nn.sigmoid(y_2)

y_3 = tf.matmul(y_r2, W_3) + b_3
y_r3 = tf.nn.sigmoid(y_3)

y_4 = tf.matmul(y_r3, W_4) + b_4
y = tf.nn.sigmoid(y_4)

#최종 결과 값 y

print("hyp shape :", np.shape(y))


BATCH_SIZE = 1000
TRAIN_SIZE = 100000


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = "./model/model.ckpt"
    saver.restore(sess, save_path)

    #검증
    start = time.process_time()

    x_final = sess.run(y, feed_dict={x: x_train_data})
    # y_final = sess.run(y_, feed_dict={x: x_train_data, y_: y_train_data})

    y_train_data = np.multiply(y_train_data, DOWNSIZE)
    x_final = np.multiply(x_final, DOWNSIZE)

    # y_final = np.multiply(y_final, DOWNSIZE)
    print(np.array(x_final, dtype='int32'))
    print(np.shape(x_final))
    print("g : ",np.array(x_final[:,1], dtype='int32'))
    res_bt = np.array(x_final[:,0], dtype='int32')
    res_gt = np.array(x_final[:,1], dtype='int32')
    res_rt = np.array(x_final[:,2], dtype='int32')

    res_b = np.reshape(res_bt,(1,-1))
    res_g = np.reshape(res_gt,(1,-1))
    res_r = np.reshape(res_rt,(1,-1))

    res_pbt = np.array(y_train_data[:,0], dtype='int32')
    res_pgt = np.array(y_train_data[:,1], dtype='int32')
    res_prt = np.array(y_train_data[:,2], dtype='int32')

    res_pb = np.reshape(res_pbt,(1,-1))
    res_pg = np.reshape(res_pgt,(1,-1))
    res_pr = np.reshape(res_prt,(1,-1))

    elapsed = (time.process_time() - start)
    print("CPU time : ",elapsed)
    print(y_train_data)
    np.savetxt('res_b.txt', res_b, delimiter=" ", fmt="%s")
    np.savetxt('res_g.txt', res_g, delimiter=" ", fmt="%s")
    np.savetxt('res_r.txt', res_r, delimiter=" ", fmt="%s")
    np.savetxt('res_pb.txt', res_pb, delimiter=" ", fmt="%s")
    np.savetxt('res_pg.txt', res_pg, delimiter=" ", fmt="%s")
    np.savetxt('res_pr.txt', res_pr, delimiter=" ", fmt="%s")

    print("테스트 완료")

    for i in range(10):
        batch_mask = np.random.choice(np.shape(x_train_data)[0], 10)  # 데이터 사이즈만큼 랜덤으로 추출
        batch_test_xs = np.array(x_train_data[batch_mask], dtype='float32')
        batch_test_ys = np.array(y_train_data[batch_mask], dtype='float32')
        batch_check = np.array(x_train_data[batch_mask], dtype='float32')

        #검증
        #랜덤하게 10개 추출
        '''
        x_final = sess.run(y, feed_dict={x: batch_test_xs})
        x_final = np.multiply(x_final, DOWNSIZE)
        BC_final = np.multiply(batch_check, DOWNSIZE)
        print("P'\n", np.array(x_final, dtype='int32'))
        print("P\n", batch_test_ys)
        print("BC\n", np.array(BC_final, dtype='int32'))
        '''
        
        #순서대로 추출
        x_final = sess.run(y, feed_dict={x: x_train_data})
        x_final = np.multiply(x_final, DOWNSIZE)
        BC_final = np.multiply(x_train_data, DOWNSIZE)
        print("P'\n", np.array(x_final[:15], dtype='int32'))
        print("BC\n", np.array(BC_final[:15], dtype='int32'))

        input()


