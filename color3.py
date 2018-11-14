#BC->P 모델 학습용

import tensorflow as tf
import sys
import numpy as np
import os

DOWNSIZE = 255

def RGBtoXYZ(r,g,b):
    r = r/DOWNSIZE
    g = g/DOWNSIZE
    b = b/DOWNSIZE

    if r>0.04045:
        r = pow(((r+0.055)/1.055),2.4)
    else:
        r = r /12.92
    if g > 0.04045:
        g = pow(((g + 0.055) / 1.055), 2.4)
    else:
        g = g / 12.92
    if b > 0.04045:
        b = pow(((b + 0.055) / 1.055), 2.4)
    else:
        b = b / 12.92
    r *= 100
    g *= 100
    b *= 100
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b  # R 을 X로 변환
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b  # G 를 Y로 변환
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b  # B 를 Z로 변환
    return x,y,z

def XYZtoLAB(x,y,z):
    x = x / 95.047          #ref_X
    y = y / 100.0           #ref_Y
    z = z / 108.883         #ref_Z

    if x > 0.008856:
        x = pow(x,(1/3))
    else:
        x = (7.787 * x ) + ( 16 / 116 )
    if y > 0.008856:
        y = pow(y,(1/3))
    else:
        y = (7.787 * y ) + ( 16 / 116 )
    if z > 0.008856:
        z = pow(z,(1/3))
    else:
        z = (7.787 * z ) + ( 16 / 116 )
    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    return l, a, b

def LABtoXYZ(l,a,b):
    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    if pow(x,3) > 0.008856:
        x = pow(x,3)
    else:
        x = (x - 16 / 116) / 7.787
    if pow(y,3) > 0.008856:
        y = pow(y,3)
    else:
        y = (y - 16 / 116) / 7.787
    if pow(z,3) > 0.008856:
        z = pow(z,3)
    else:
        z = (z - 16 / 116) / 7.787
    x = x * 95.047          #ref_X
    y = y * 100.0           #ref_Y
    z = z * 108.883         #ref_Z
    return x,y,z

def XYZtoRGB(x,y,z):
    x = x/100
    y = y/100
    z = z/100

    r = 3.2406 * x + -1.5372 * y + -0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0414 * z
    b = 0.0557 * x + -0.2040 * y + 1.0570 * z

    if r > 0.0031308:
        r = 1.055 * (pow(r,(1/2.4))) - 0.055
    else:
        r = 12.92 * r
    if g > 0.0031308:
        g = 1.055 * (pow(g,(1/2.4))) - 0.055
    else:
        g = 12.92 * g
    if b > 0.0031308:
        b = 1.055 * (pow(b,(1/2.4))) - 0.055
    else:
        b = 12.92 * b
    r = r * 255
    g = g * 255
    b = b * 255
    return r,g,b


#DATA_NUM = 175104
"""데이터 불러오기"""
with  open("./data/data_all.txt",'r') as input_data:
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
        input_data_0[i][2],input_data_0[i][1],input_data_0[i][0] = RGBtoXYZ(input_data_0[i][2],input_data_0[i][1],input_data_0[i][0])
        y_data.extend(input_data_0[i])
    elif i%3==1:        #B_data
        input_data_0[i][2],input_data_0[i][1],input_data_0[i][0] = RGBtoXYZ(input_data_0[i][2],input_data_0[i][1],input_data_0[i][0])
        x_data.extend(input_data_0[i])
    else:               #C_data
        input_data_0[i][2],input_data_0[i][1],input_data_0[i][0] = RGBtoXYZ(input_data_0[i][2],input_data_0[i][1],input_data_0[i][0])
        x_data.extend(input_data_0[i])

print("y data : ", len(y_data))
#Train, Test, Validation 데이터 나누기 85%, 10%, 5%로 나눔
DATA_NUM = int(len(y_data)/3)
DATA_TRAIN_NUM = int(DATA_NUM*0.85)
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
x_test_data_ = np.array(x_data[(DATA_TRAIN_NUM*3*2):-(DATA_VAL_NUM*3*2)], dtype='float32')

y_train_data_ = np.array(y_data[:(DATA_TRAIN_NUM*3)], dtype='float32')
y_test_data_ = np.array(y_data[(DATA_TRAIN_NUM*3):-(DATA_VAL_NUM*3)], dtype='float32')

#데이터 shape 변경
x_train_data = np.reshape(x_train_data_, (-1,6))
x_test_data = np.reshape(x_test_data_, (-1,6))
y_train_data = np.reshape(y_train_data_, (-1,3))
y_test_data = np.reshape(y_test_data_, (-1,3))
print(x_train_data[:10])


#색상 데이터를 255로 나눠줘 0~1로 normalize
x_train_data = np.divide(x_train_data,DOWNSIZE)
x_test_data = np.divide(x_test_data,DOWNSIZE)
y_train_data = np.divide(y_train_data,DOWNSIZE)
y_test_data = np.divide(y_test_data,DOWNSIZE)


print(np.shape(y_train_data))

"""모델 설계"""


x = tf.placeholder(tf.float32, [None,6])
y_ = tf.placeholder(tf.float32, [None,3])

keep_prob = tf.placeholder(tf.float32)

W_1 = tf.Variable(tf.random_uniform([6,50],-1.0,1.0), name= 'W_1')
b_1 = tf.Variable(tf.random_uniform([50],-1.0,1.0), name= 'b_1')

W_2 = tf.Variable(tf.random_uniform([50,100],-1.0,1.0), name= 'W_2')
b_2 = tf.Variable(tf.random_uniform([100],-1.0,1.0), name= 'b_2')

W_3 = tf.Variable(tf.random_uniform([100,150],-1.0,1.0), name= 'W_3')
b_3 = tf.Variable(tf.random_uniform([150],-1.0,1.0), name= 'b_3')

W_4 = tf.Variable(tf.random_uniform([150,3],-1.0,1.0), name= 'W_4')
b_4 = tf.Variable(tf.random_uniform([3],-1.0,1.0), name= 'b_4')

#W_5 = tf.Variable(tf.random_uniform([10,3],-1.0,1.0), name= 'W_5')
#b_5 = tf.Variable(tf.random_uniform([3],-1.0,1.0), name= 'b_5')

y_1 = tf.matmul(x,W_1) + b_1
y_r1 = tf.nn.sigmoid(y_1)
y_r1 = tf.nn.dropout(y_r1,keep_prob=keep_prob)

y_2 = tf.matmul(y_r1, W_2) + b_2
y_r2 = tf.nn.sigmoid(y_2)
y_r2 = tf.nn.dropout(y_r2,keep_prob=keep_prob)

y_3 = tf.matmul(y_r2, W_3) + b_3
y_r3 = tf.nn.sigmoid(y_3)
y_r3 = tf.nn.dropout(y_r3,keep_prob=keep_prob)

y_4 = tf.matmul(y_r3, W_4) + b_4
y = tf.nn.sigmoid(y_4)
y = tf.nn.dropout(y,keep_prob=keep_prob)


print("hyp shape :", np.shape(y))
print("y shape :", np.shape(y_))

###Cost
cost_2 = tf.reduce_sum(tf.abs(y - y_))
cost = tf.reduce_mean(tf.square(y-y_))
global_step = tf.Variable(0,trainable=False)
BATCH_SIZE = 1000
TRAIN_SIZE = 80000

###Learning Rate
boundaries = [40000, 50000, 60000, 70000, 80000, 90000, 100000, 600000, 650000, 700000, 750000, 80000] #12
values = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.007, 0.005, 0.003, 0.001, 0.0005 ]       #boundaries 보다 +1개 13
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

###Optimizer
optimizer = tf.train.AdadeltaOptimizer(learning_rate)
train = optimizer.minimize(cost,global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    ###체크포인트 불러옴
    #load_path = "./model/model.ckpt"
    #saver.restore(sess, load_path)

    #총 데이터를 10000번 학습시킴 이부분에서 배치를 써줘야 Loss가 정상적으로 줄어들듯
    for step in range(TRAIN_SIZE):
        batch_mask = np.random.choice(np.shape(x_train_data)[0], BATCH_SIZE) #데이터 사이즈만큼 랜덤으로 추출
        batch_xs = x_train_data[batch_mask]
        batch_ys = y_train_data[batch_mask]

        if step % 5000 == 0:
            print("step : ", step, "cost : ", sess.run(cost, feed_dict={x: x_test_data, y_: y_test_data, keep_prob:1}),
                  "learning rate : ", sess.run(learning_rate, {x: batch_xs, y_: batch_ys}))
            param_list = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]
            saver = tf.train.Saver(param_list)
            save_path = saver.save(sess, "./model/model.ckpt")
            print(os.getcwd())
            print("Model saved in file: ", save_path)
         #dropout keep_prob 조정
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:1})

    print("step : ", step, "cost : ", sess.run(cost, feed_dict={x: x_test_data, y_: y_test_data, keep_prob:1}),
          "learning rate : ", sess.run(learning_rate, {x: x_test_data, y_: y_test_data, keep_prob:1}))
    save_path = saver.save(sess, "./model/model.ckpt")
    print(os.getcwd())
    print("Model saverd in file: ", save_path)

    #학습된 신경망을 테스트하기위해 test 데이터를 곱해줌
    for i in range(10):
        batch_mask = np.random.choice(np.shape(x_train_data)[0], 10)  # 데이터 사이즈만큼 랜덤으로 추출
        batch_test_xs = np.array(x_train_data[batch_mask], dtype='float32')
        batch_test_ys = np.array(y_train_data[batch_mask], dtype='float32')
        batch_check = np.array(x_train_data[batch_mask], dtype='float32')

        #검증
        x_final = sess.run(y, feed_dict={x: batch_test_xs, y_: batch_test_ys, keep_prob:1})
        y_final = sess.run(y_, feed_dict={x: batch_test_xs, y_: batch_test_ys, keep_prob:1})
        x_final = np.multiply(x_final, DOWNSIZE)
        y_final = np.multiply(y_final, DOWNSIZE)
        BC_final = np.multiply(batch_check, DOWNSIZE)

        #LAB to RGB 로 변환
        for i in range(10):
            x_final[i][2], x_final[i][1], x_final[i][0] = XYZtoRGB(x_final[i][2], x_final[i][1], x_final[i][0])
            y_final[i][2], y_final[i][1], y_final[i][0] = XYZtoRGB(y_final[i][2], y_final[i][1], y_final[i][0])
        for i in range(10):
            for j in range(2):
                BC_final[i][j * 3 + 2], BC_final[i][j * 3 + 1], BC_final[i][j * 3] = XYZtoRGB(BC_final[i][j * 3 + 2], BC_final[i][j * 3 + 1], BC_final[i][j * 3])
        print("P'\n", np.array(x_final, dtype='int32'))
        print("P\n", np.array(y_final, dtype='int32'))
        print("BC\n", np.array(BC_final, dtype='int32'))

        print("reduce_sum(abs(A-C)) : ",sess.run(cost_2, feed_dict={x:x_train_data, y_: y_train_data, keep_prob:1}))

        input()
