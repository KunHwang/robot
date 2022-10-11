import tensorflow as tf
import gym
from IPython import display #Public API for display tools in IPython
#from PIL import Image #PIL : 이미지 분석 및 처리를 쉽게 할 수 있는 라이브러리(Python Imaging Library : PIL)가 있습니다. 바로 pillow모듈입니다
#display.display(Image.open('sample.jpeg')) #PIL모듈을 사용하여 객체를 생성하지 않고 그림을 표시하기 위해 하위 패키지Image를 가져올 수 있습니다.
#lst = np.arange(10)
#print(lst)
#display.display(lst)

import cv2

import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt #matplotlib를 통해서도 image를 display할 수 있다. 
#import matplotlib.image as mpimg
#img = mpimg.imread('sample.jpeg')
#imgplot = plt.imshow(img)
#plt.show()             

from collections import deque
import numpy as np
import random

#print() 대신 IPython.display를 사용할 수 있다
#from IPython import display # lst = np.arange(10); display(lst)
%matplotlib inline

# 동영상으로 저장하기 위한 이미지 를 그려주는 함수를 정의하자
def draw_state(img, q_val):
    img_r = img.copy()
    cv2.putText(img_r, 'L', (20, 50), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    cv2.putText(img_r, 'R', (530, 50), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    cv2.putText(img_r, str(round(q_val[0], 3)), (10, 70), 
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img_r, str(round(q_val[1], 3)), (520, 70), 
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    return img_r


# 문자열을 이미지로 그려주는 함수를 정의하자
def draw_txt(txt):
    img = np.zeros((400, 600, 3))
    cv2.putText(img, txt, (10, 200), 
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))
    return img

# 상태(3가지값: )를 입력 받으면, 행동을 결정하고, Q값을 예측하는 신경망을 모델링하자
state_num = (3,) # 상태: angle(theta)/angular velocity(theta_dot) -> theta decomposed by cos(theta),sin(theta), and theta_dot 
action_num = 1 # 연속적인 행동: [-2, 2]
# 보상: reward = - cost , 0 일때 max 
hidden_state = 128
#learning_rate = 0.001
learning_rate = 0.0001 #학습율을 높였을 때 env에서 error 발생. 모델 파라메터가 너무 급격히 업데이트 되다보니 모델에서 action을 제대로 예측 못하서 이값이 env에 들어가니까 env 에러가 발생   

i=tf.keras.Input(shape=state_num,name='state_in')
out=tf.keras.layers.Dense(hidden_state,activation='relu')(i)
#out=tf.keras.layers.BatchNormalization()(out)

mu=tf.keras.layers.Dense(hidden_state//2,activation='relu')(out) # // (floar division) 는 나눗셈을 의미하며 결과가 int 로 나타납니다.
mu=tf.keras.layers.Dense(hidden_state//4,activation='relu')(mu)
mu=tf.keras.layers.Dense(action_num,activation='tanh',name='mu')(mu) #tanh(tangent hyperbolic)로 -1 ~ 1사이를 리턴
mu=tf.keras.layers.Lambda(lambda x : x*2.0)(mu) # x[-1~1] 값이 출력되면 [-2~2]로 만듬
#tf.keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None, **kwargs)
#layers.Lambda : 케라스는 패키지에서 제공하지 않는 새로운 인공신경망의 기능을 추가하는 다양한 방법을 제공합니다. 가장 간단한 방법은 Lambda 클래스를 이용하는 겁니다. 이 클래스를 이용하면 새로운 계층을 파이썬의 lambda 함수처럼 간단하게 만들어 사용할 수 있다.
#lamda 함수: lambda 매개변수 : 표현식 --> plus_ten = lambda x: x + 10 #  plus_ten(1) = 11

#sigma는 무조건 양수 0 ~ 1까지. 0이 나오면 Pdf가 너무 좁아지기 때문에 이를 방지하기 위해 0.1 추가하여 pdf를 곡선을 스무딩한다.
sigma=tf.keras.layers.Dense(hidden_state//2,activation='relu')(out)
sigma=tf.keras.layers.Dense(hidden_state//4,activation='relu')(sigma)
sigma=tf.keras.layers.Dense(action_num,activation='sigmoid',name='sigma')(sigma)
sigma=tf.keras.layers.Lambda(lambda x : x+0.1)(sigma)

v_out = tf.keras.layers.Dense(hidden_state//2, activation='relu')(out)
v_out = tf.keras.layers.Dense(hidden_state//4, activation='relu')(v_out)
v_out = tf.keras.layers.Dense(1,name='v_out')(v_out)

#매상태에 마다 action에 대한 정규분포가 출력된다. 
model=tf.keras.Model(inputs=[i],outputs=[mu,sigma,v_out])

opt = tf.keras.optimizers.Adam(learning_rate)
model.summary()

from tensorflow.keras.utils import plot_model
#plot_model(model, show_shapes=False)
plot_model(model, show_shapes=True)

import tensorflow_probability as tfp

# 파라미터들을 설정해 주자
num_episode = 500
batch_size=64 # n-step (걸음수)를 64번 까지 설정한다. : 64 step 마다 policy를 업데이트한다. 
discount_rate = 0.99
gae_rate = 0.9 # generalized advantage estimation : 걸음수 마다 비중을 떨어뜨리는 비율 (델타) ~ discount rate와 유사
ppo_epoch = 10 # 새로운 나와 과거의 나가 같을때 까지 계속 반복해주는 횟수 (엑기스를 뽑기 위해 여러번 짜내는 횟수)
ppo_esp = 0.2 # 새로운 나와 과거의 나가 같은 지를 판단하는 기준 : 20% 차이 안에서는 같은 나라고 판단함. 

is_video_save = False
fps = 5.0
avi_file_name = 'CA-Pendulum-REINFORCE.avi'

# 환경을 만들어 주자
env = gym.make('Pendulum-v0')

reward_list = []

for epi in range(1, num_episode+1):
    d = False
    total_reward = 0
    s = env.reset()
    s = np.reshape(s, [1,3])

    batch_state=[]
    batch_action=[]
    batch_reward=[]
    batch_old_p=[]

    while not d:
        m, si, val = model.predict(s, verbose=0)
        dist = tfp.distributions.Normal(loc=m[0], scale=si[0])
        action = dist.sample()
        action = tf.clip_by_value(action, -2.0, 2.0)
        p = dist.prob(action)
        n_s, r, d, _ = env.step(action)
        n_s = np.reshape(n_s, [1, 3])
        scale_r = r/ 1000.

        batch_state.append(s)
        batch_action.append(np.reshape(action, (1,1)))
        batch_reward.append(np.reshape(scale_r,(1,1)))
        batch_old_p.append(np.reshape(p,(1,1)))

        if (len(batch_state) == batch_size) or d: #배치사이즈가 차거나 또는 게임이 끝나면 학습을 한다. 
            #batch를 넣어준 걸 꺼낸다.
            batch_state = np.reshape(batch_state, (-1, 3))
            batch_action = np.reshape(batch_action, (-1,1))
            batch_reward = np.reshape(batch_reward, (-1,1))
            batch_old_p = np.reshape(batch_old_p, (-1,1))

            for ppo_e in range(ppo_epoch):
                with tf.GradientTape() as tape:
                    _m, _si, _val = model(batch_state)
                    _n_m, _n_si, _n_val = model(n_s)
                    _dist = tfp.distributions.Normal(loc=_m, scale=_si)
                    #_dist = tfp.distributions.Normal(loc=_m[0], scale=_si[0])
                    _p = _dist.prob(batch_action)

                    td = np.zeros_like(batch_reward) #입력과 같은 shape의 zero array 리턴
                    gae = np.zeros_like(batch_reward)
                    gae_sum = 0
                    r_sum = 0
                    r_sum = tf.stop_gradient(_n_val[0])
                    td_sum = tf.stop_gradient(_n_val[0])
                    #GAE를 계산하는 수식
                    for i in reversed(range(len(batch_reward))):
                        delta = batch_reward[i] + discount_rate * r_sum - tf.stop_gradient(_val[i])
                        gae_sum = discount_rate * gae_rate * gae_sum + delta
                        td_sum = batch_reward[i] +discount_rate * td_sum
                        gae[i] = gae_sum
                        r_sum = tf.stop_gradient(_val[i])
                        td[i] = td_sum

                    tde = td - _val

                    ratio = tf.math.exp(tf.math.log(_p+0.001)-tf.math.log(batch_old_p+0.001))
                    s_1 = ratio * gae
                    s_2 = tf.clip_by_value(ratio, 1.0 - ppo_esp, 1.0 + ppo_esp)*gae
                    p_loss = -tf.math.minimum(s_1, s_2)
                    v_loss = tf.square(tde) * 10.0
                    loss = p_loss * v_loss
                grad = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grad, model.trainable_variables))

            batch_state = []
            batch_action = []
            batch_reward = []
            batch_old_p = []

        s = n_s
        total_reward = total_reward + r

    reward_list.append(total_reward)

    print('현재 에피소드: {}, 현재 점수: {}, 30번 평균: {}'.format(epi, total_reward, np.mean(reward_list[-30:])))


plt.plot(reward_list)
plt.savefig('plot.jpg')
if (is_video_save):
    txt = 'Total Score : '+str(np.sum(reward_list))
    for _ in range(3):
        out.write(np.uint8(draw_txt(txt)))
    reward_list_img = cv2.imread('plot.jpg')
    reward_list_img = cv2.resize(reward_list_img, (600, 400))
    for _ in range(10):
        out.write(np.uint8(reward_list_img))
    out.release()
