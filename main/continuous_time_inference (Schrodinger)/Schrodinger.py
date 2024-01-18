"""
@author: Maziar Raissi
"""

#下面这行代码，是为了把自己编写的代码文件当作一共模块导入，这里是把Utilities文件夹中的plotting.py文件当作python的模块导入，对应的是下面的from plotting import newfig, savefig。路径要随着不同设备的系统做相应的修改
import sys #导入sys模块。sys模块提供了一些变量和函数，用于与 Python解释器进行交互和访问。例如，sys.path 是一个 Python 在导入模块时会查找的路径列表，sys.argv 是一个包含命令行参数的列表，sys.exit() 函数可以用于退出 Python 程序。导入 sys 模块后，你就可以在你的程序中使用这些变量和函数了。
sys.path.insert(0, 'C:/Users/cheny/Documents/GitHub/PINNs/Utilities/') #在 Python的sys.path列表中插入一个新的路径。sys.path是一个 Python 在导入模块时会查找的路径列表。新的路径'../../Utilities/'相对于当前脚本的路径。当你尝试导入一个模块时，Python 会在 sys.path 列表中的路径下查找这个模块。通过在列表开始位置插入一个路径，你可以让 Python 优先在这个路径下查找模块。这在你需要导入自定义模块或者不在 Python 标准库中的模块时非常有用。

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#下面的`scipy`是一个用于科学计算和技术计算的Python库，提供了许多高级的数学函数和便利的操作，包括数值积分、插值、优化、图像处理、统计等。
import scipy.io #导入了scipy库中的io模块。scipy.io模块包含了一些用于文件输入/输出的函数，例如读取和写入.mat文件（MATLAB格式）。
from scipy.interpolate import griddata#`scipy.interpolate`是`scipy`库中的一个模块，提供了许多插值工具，用于在给定的离散数据点之间进行插值和拟合。`griddata`是这个模块中的一个函数，用于在无规则的数据点上进行插值。它使用方法如下：
#griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)；
   # `points`： ndarray of floats, shape (n, D)。表示数据点的坐标。`values`： ndarray of float or complex, shape (n,)。表示数据点的值。`xi`： ndarray of float, shape (M, D)。表示插值点的坐标。`method`： 插值方法，可选'linear'、'nearest'、'cubic'。默认为'linear'。
   #`fill_value`： 在插值范围外的点的值。默认为nan。`rescale`： 是否对坐标点进行重标定，以提高数值稳定性。默认为False。
   #返回值：ndarray，shape (M,) or (M, 1)。插值点的值。这个函数可以用于从散列的数据点创建一个连续的函数，这对于处理实际数据非常有用，因为实际数据通常是不规则或者不完整的。
from pyDOE import lhs #`pyDOE`是一个Python库，用于设计实验。它提供了一些函数来生成各种设计，如因子设计、拉丁超立方设计等。`lhs`是库中的一个函数，全名为"Latin Hypercube Sampling"，拉丁超立方采样。这是一种统计方法，用于生成一个近似均匀分布的多维样本点集。它在参数空间中生成一个非常均匀的样本，这对于高维数值优化问题非常有用，因为它可以更好地覆盖参数空间。
#`lhs`函数的基本用法如下：lhs(n, samples=1000):其中，`n`是参数的数量，`samples`是想生成的样本点的数量。这个函数会返回一个形状为(samples, n)的数组，每一行都是一个n维的样本点，所有的样本点都在[0, 1]范围内。
from plotting import newfig, savefig #从自定义的plotting.py文件中导入了newfig和savefig函数。这两个函数用于创建和保存图形。这两个函数的定义在plotting.py文件中
from mpl_toolkits.mplot3d import Axes3D #`mpl_toolkits.mplot3d`是`matplotlib`库的一个模块，用于创建三维图形。`Axes3D`是`mpl_toolkits.mplot3d`模块中的一个类，用于创建一个三维的坐标轴。可以在这个坐标轴上绘制三维的图形，如曲线、曲面等。
import time #一个内置模块，用于处理时间相关的操作。
import matplotlib.gridspec as gridspec #是`matplotlib`库的一个模块，用于创建一个网格布局来放置子图。在`matplotlib`中可以创建一个或多个子图（subplot），每个子图都有自己的坐标轴，并可以在其中绘制图形。`gridspec`模块提供了一个灵活的方式来创建和放置子图。
from mpl_toolkits.axes_grid1 import make_axes_locatable #`mpl_toolkits.axes_grid1`是`matplotlib`库的一个模块，提供了一些高级的工具来控制matplotlib图形中的坐标轴和颜色条。`make_axes_locatable`是模块中的一个函数，用于创建一个可分割的坐标轴。可以在这个坐标轴的四个方向（上、下、左、右）添加新的坐标轴或颜色条。

#下面两行是指定索引为哪一块的gpu进行训练，这里使用索引为1的，即第二块gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1，0"


#NumPy和TensorFlow都有自己的随机数生成器，它们是独立的，互不影响。也就是说，设置NumPy的随机数种子不会影响TensorFlow的随机数生成，反之亦然
np.random.seed(1234) #设置了NumPy的随机数生成器的种子。设置随机数生成器的种子可以确保每次运行程序时，NumPy生成的随机数序列都是一样的。
tf.set_random_seed(1234) #设置了TensorFlow的随机数生成器的种子。设置随机数生成器的种子可以确保每次运行程序时，TensorFlow生成的随机数序列都是一样的



#定义了一个名为`PhysicsInformedNN'的类，用于实现基于物理的神经网络。
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub): #这个类包含的第一个方法__init__，这是一个特殊的方法，也就是这个类的构造函数，用于初始化新创建的对象，接受了几个参数
        
        
        #`numpy.concatenate`是一个用于数组拼接的函数。它可以将多个数组沿指定的轴拼接在一起，形成一个新的数组：numpy.concatenate((a1,a2, ...), axis=0)其中，`a1,a2, ...`是需要拼接的数组（只能接受数组或序列类型的参数，且参数形状必须相同），可以是多个。`axis`参数用于指定拼接的轴向，`axis=0`表示沿着第一个轴（即行）进行拼接，不指定`axis`参数默认值是0。
        X0 = np.concatenate((x0,0*x0), 1) # [x0, 0],将x0和0*x0两个数组在第二个维度（即列）上进行了合并。0*x0会生成一个与x0形状相同，但所有元素都为0的数组。因此，X0的结果是一个新的二维数组，其中第一列是x0的值，第二列全为0
        X_lb = np.concatenate((0*tb+lb[0],tb), 1) # [lb[0], tb],将0*tb+lb[0]和tb两个数组在第二个维度（即列）上进行了合并。0*tb+lb[0]会生成一个与tb形状相同，但所有元素都为lb[0]的数组。因此，X_lb的结果是一个新的二维数组，其中第一列全为lb[0]的值，第二列是tb的值。
        X_ub = np.concatenate((0*tb+ub[0],tb), 1) # [ub[0], tb],同上生成一个与tb形状相同，但所有元素都为ub[0]的数组。因此，X_ub的结果是一个新的二维数组，其中第一列全为ub[0]的值，第二列是tb的值
        
        #Python使用self关键字来表示类的实例。当在类的方法中定义一个变量时，例如lb和ub，这些变量只在该方法内部可见，也就是说它们的作用域仅限于该方法。当方法执行完毕后，这些变量就会被销毁，无法在其他方法中访问它们。但如果希望在类的其他方法中也能访问这些变量就需要将它们保存为类的实例属性。这就是self.lb和self.ub的作用。
            #通过将lb和ub赋值给self.lb和self.ub，就可以在类的其他方法中通过self.lb和self.ub来访问这些值。总的来说，self.lb和self.ub是类的实例属性，它们的作用域是整个类，而不仅仅是定义它们的方法。
        self.lb = lb #将传入的lb和ub参数的值存储在实例中，以便后续使用。这样可以在类的其他方法中通过self.lb和self.ub来访问这些值。
        self.ub = ub
               
        self.x0 = X0[:,0:1] #将X0的第一列赋值给self.x0（:表示取所有行,0：1实际上表示取第一列，因为python是左闭右开的）,将X0的第二列赋值给self.t0。这样可以在类的其他方法中通过self.x0和self.t0来访问这些值。
        self.t0 = X0[:,1:2] #将x0的第二列赋值给self.t0

        self.x_lb = X_lb[:,0:1] #将X_lb的第一列赋值给self.x_lb
        self.t_lb = X_lb[:,1:2] #将X_lb的第二列赋值给self.t_lb

        self.x_ub = X_ub[:,0:1] #将X_ub的第一列赋值给self.x_ub
        self.t_ub = X_ub[:,1:2] #将X_ub的第二列赋值给self.t_ub
        
        self.x_f = X_f[:,0:1] #将X_f的第一列赋值给self.x_f
        self.t_f = X_f[:,1:2] #将X_f的第二列赋值给self.t_f
        
        self.u0 = u0 #将传入的u0和v0参数的值存储在实例中，以便后续使用。这样可以在类的其他方法中通过self.u0和self.v0来访问这些值。
        self.v0 = v0
        
        # Initialize NNs 
        self.layers = layers #将传入的layers参数的值存储在实例中，以便后续使用。这样可以在类的其他方法中通过self.layers来访问这些值。
        self.weights, self.biases = self.initialize_NN(layers) #调用了initialize_NN方法，用于初始化神经网络的权重和偏置。这个方法接受一个参数layers，它是一个列表，包含了神经网络的层数和每一层的神经元数量。例如，layers=[2, 100, 100, 100, 100, 2]表示神经网络有5个隐藏层，每个隐藏层有100个神经元，输入层和输出层分别有2个神经元。这个方法返回了神经网络的权重和偏置（具体见下面），分别存储在self.weights和self.biases中。
        
        #使用TensorFlow库创建占位符，用于存储输入和输出数据。占位符是TensorFlow中的一个特殊对象，允许在运行时将数据传递给TensorFlow计算图。可以将占位符看作是一个变量，但不需要提供初始值。相反只需要在运行计算图时提供一个值。使用feed_dict参数来为占位符提供值。
            #例如，如果x是一个占位符，可以使用feed_dict={x: 1.0}来为它提供值。这个值可以是一个单独的数字，也可以是一个数组。        
        #每个占位符都使用了tf.placeholder函数进行创建。该函数接受两个参数,第一个是数据类型,这里都是tf.float32,表示占位符的数据类型是浮点数。第二个参数是形状,这里都是[None,self.x0.shape[1]]这样的形式，其中None表示这一维的长度可以是任意的，self.x0.shape[1]表示这一维的长度是self.x0的列数。
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])



        # tf Graphs，这里是使用TensorFlow库进行神经网络前向传播的部分。
        self.u0_pred, self.v0_pred, _ , _ = self.net_uv(self.x0_tf, self.t0_tf) #是调用net_uv函数,将self.x0_tf和self.t0_tf作为参数传入,然后将返回的前两个结果赋值给self.u0_pred和self.v0_pred。后两个_是Python惯用法，表示不关心net_uv函数返回的后两个结果。
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf) #同上，不过这里函数返回的后两个结果会赋值给self.u_x_lb_pred和self.v_x_lb_pred。
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf) #同上
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf) #调用net_f_uv函数,将self.x_f_tf和self.t_f_tf作为参数传入,然后将返回的结果赋值给self.f_u_pred和self.f_v_pred。




        # Loss，这里是使用TensorFlow库计算损失函数的部分，训练目标是最小化损失函数，这里的损失函数由八部分组成，分别是初始条件、边界条件、微分方程两边的残差。每一部分都是预测值与真实值之间的差的平方的均值（均方误差）
        #tf.reduce_mean是TensorFlow库中的一个函数，用于计算张量的均值。它接受一个参数，即张量，可以是一个一维数组，也可以是一个多维数组。它会返回一个标量，即这个张量的均值。
        #tf.square是TensorFlow库中的一个函数，用于计算张量的平方。它接受一个参数，即张量，可以是一个一维数组，也可以是一个多维数组。它会返回一个与输入张量形状相同的张量，其中每个元素都是输入张量对应元素的平方。
        #这里的+ \表示将两行代码连接成一行，这是Python中的行连接符，用于将一行代码分成多行书写。
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))



        # Optimizers，这里是使用TensorFlow库进行优化的部分，使用了两种优化器，分别是L-BFGS-B和Adam。L-BFGS-B是一种基于梯度的优化方法，它使用了拟牛顿法来寻找损失函数的最小值。Adam是一种基于梯度的优化方法，它使用了自适应学习率来寻找损失函数的最小值

        #首先用tf.contrib.opt.ScipyOptimizerInterface函数创建了一个优化器self.optimizer，它使用了L-BFGS-B方法，最大迭代次数为50000次，最大函数调用次数为50000次，最大相关矩阵大小为50，最大线搜索次数为50，终止条件为1.0 * np.finfo(float).eps。
              #tf.contrib.opt.ScipyOptimizerInterface是TensorFlow中的函数，提供了一个接口，可以使用优化算法来最小化TensorFlow的损失函数，接受三个参数，第一个参数是损失函数（这里是self.loss）,第二个参数是优化方法，这里是L-BFGS-B，第三个参数options是一个字典，用于指定优化器的参数。
              #这里maxiter表示最大迭代次数，maxfun表示最大函数调用次数，maxcor表示每次迭代中使用的最大修正因子数量，maxls表示每次迭代中最大线搜索次数，ftol表示终止条件，这里是1.0*np.finfo(float).eps，其中np.finfo(float).eps表示浮点数的精度，1.0 * np.finfo(float).eps表示浮点数的精度乘以1.0。
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps}) #np.finfo()是numpy库中的一个函数，用于获取浮点数的精度。np.finfo(float).eps返回的是浮点数的及其精度，也就是1.0和比1.0大的最小浮点数之间的差(大多数机器上，对于双精度浮点数，如python中的float，这个值大约是2.22e-16)
        #使用函数tf.train.AdamOptimizer创建了一个优化器self.optimizer_Adam，它使用了Adam方法
            #tf.train.AdamOptimizer函数语法：tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')，第一个参数表示学习率；第二个参数表示一阶矩估计的指数衰减率；第三个参数表示二阶矩估计的指数衰减率；第四个参数表示防止除0错误的小常数；第五个参数默认False，若为True则在更新操作中对变量加锁；最后一个是操作的名称，默认为'Adam'
        self.optimizer_Adam = tf.train.AdamOptimizer()
        #使用函数创建了一个名为self.train_op_Adam的训练操作，使用Adam优化器最小化损失函数（这里是之前定义的self.loss）
            #minimize函数语法：minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)，参数loss表示需要最小化的损失函数；第二个参数可选，若提供则每次优化操作后该值增加1；第三个参数是需要优化的变量列表（不提供默认所有的可训练变量）；其他参数大多数情况下不需要设置
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                



        # tf session，使用TensorFlow库创建了一个会话self.sess，是用于执行计算图的环境。
    
        #使用tf.Session创建了一个名为tf.sess的会话。接受一个config参数，这是一个tf.ConfigProto对象，用于设置会话的配置选项。这里设置了两个选项：
              #allow_soft_placement：如果设置为True，那么当某些操作无法在GPU上执行时，TensorFlow会自动将它们放在CPU上执行；
              #log_device_placement：如果设置为True，那么在日志中会记录每个节点被安排在哪个设备上执行
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        #使用tf.global_variables_initializer函数创建了一个初始化所有全局变量的操作（Tensorflow中所有变量在使用之前都需要进行初始化）
        init = tf.global_variables_initializer()
        #sess.run是会话的一个方法，用于执行图中的操作或计算张量的值。这里是执行初始化
        self.sess.run(init)

    #定义了一个名为`initialize_NN'的函数/方法，用于初始化神经网络的权重和偏置，最后返回权重和偏置。          
    def initialize_NN(self, layers): #接受一个参数layers
        #定义了两个空列表weights和biases，用于存储神经网络的权重和偏置。        
        weights = []
        biases = []
        num_layers = len(layers)  #获取神经网络的层数（由输入参数给出）并将其赋值给num_layers
        for l in range(0,num_layers-1):  #使用循环遍历神经网络的每一层（除了输出层）
            W = self.xavier_init(size=[layers[l], layers[l+1]]) #初始化该层的权重，使用了xavier_init函数(该函数定义在下面)进行Xaiver初始化，权重的形状是layers[l]*layers[l+1]，其中layers[l]是上一层的神经元数量，layers[l+1]是当前层的神经元数量。
              #tf.Variable(initial_value, dtype=None, name=None)函数用于创建一个变量，接受三个参数，第一个参数initial_value是变量的初始值，第二个参数dtype是变量的数据类型，第三个参数name是变量的名称。
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32) #初始化该层的偏置b，使用了tf.Variable函数，将偏置b初始化为一个形状为[1,layers[l+1]]的全0数组，其中layers[l+1]是当前层的神经元数量
              #list.append(element)是Python中的一个方法，用于向列表中添加元素（添加到列表末尾）。list是要添加元素的列表，element是要添加的元素（然和类型的参数均可）
            weights.append(W) #将初始化的权重和偏置添加到weights和biases列表中
            biases.append(b)        
        return weights, biases


 

    #定义了一个名为xavier_init的函数/方法，用于初始化神经网络的权重(在神经网络参数初始化中实现，见上面)。这个函数使用了Xavier初始化方法，这是一种常用的权重初始化方法，可以帮助我们在训练深度神经网络时保持每一层的激活值的分布相对稳定。
    def xavier_init(self, size):   #接受一个参数size
        in_dim = size[0]  #输入维度是size的第一个数
        out_dim = size[1]     #输出维度是size的第二个数
        xavier_stddev = np.sqrt(2/(in_dim+out_dim))   #计算标准差
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32) #返回一个变量，类型为32位浮点，初始值为截断正态分布，标准差为xavier_stddev，形状为[in_dim, out_dim]，其中in_dim和out_dim分别是输入维度和输出维度
    
    #定义了一个名为neural_net的函数/方法，用于实现神经网络的输出。这个方法接受三个参数，分别是X，weights和biases，其中X是输入数据，weights和biases是神经网络的权重和偏置。
    def neural_net(self,X,weights,biases):
        num_layers=len(weights)+1 #计算神经网络的层数并返回到num_layers，其值位权重矩阵的长度（行数）加1
        
        H=2.0*(X-self.lb)/(self.ub-self.lb)-1.0 #这里H是X经过归一化处理后的结果，将X映射到了[-1,1]区间内
        for l in range(0,num_layers-2): #使用循环遍历神经网络的每一层（除了输出层）
            W=weights[l] #获取当前层的权重
            b=biases[l] #获取当前层的偏置
            H=tf.tanh(tf.add(tf.matmul(H,W),b)) #计算当前层的输出，使用了tf.matmul函数计算矩阵乘法，tf.add函数计算矩阵加法，tf.tanh函数计算双曲正切函数
        W=weights[-1] #获取输出层的权重,这里的weights[-1]表示列表weights的最后一个元素
        b=biases[-1] #获取输出层的偏置
        Y=tf.add(tf.matmul(H,W),b)  #计算输出层的输出H*W+b
        return Y #返回输出层的输出Y
    


    #定义了一个名为net_uv的函数/方法，用于计算神经网络的输出以及输出关于输入x的梯度。这个方法接受两个参数，分别是x和t，其中x是输入数据，t是时间数据。最后返回神经网络的两个输出以及输出它们关于输入x的梯度。
    def net_uv(self,x,t):
            #tf.concat(values, axis)，用于将多个张量在指定的维度上进行拼接，接受两个参数，第一个参数values是一个列表，表示需要拼接的张量；第二个参数axis是一个整数，表示拼接的维度。
        X=tf.concat([x,t],1)   #将输入的两个参数x和t在第二个维度（列）上进行拼接，形成一个新的张量X
        #调用之前定义的neural_net函数，根据两个参数权重和偏置，以及上一步得到的张量X，计算神经网络的输出uv
        uv=self.neural_net(X,self.weights,self.biases)
        #将uv（是一个二维张量）的第一列赋值给u，第二列赋值给v
        u=uv[:,0:1]
        v=uv[:,1:2]
        #分别计算u和v关于x的梯度，使用了tf.gradients函数，它接受两个参数，第一个参数ys是一个张量，表示需要求导的张量；第二个参数xs是一个张量或张量列表，表示需要对哪些张量求导。
            #tf.gradients(ys, xs, grad_ys=None, name="gradients", colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)：第一个参数ys是一个张量，表示需要求导的张量；第二个参数xs是一个张量或张量列表，表示需要对哪些张量求导；
            #第三个参数表示一个与ys相同长度的张量列表，每个张量都是ys中对应张量的梯度，若为None则假设每个ys的梯度为1；其他参数用于控制梯度计算的细节，通常不需要修改
            #最后的输出是一个列表，列表中的每个元素都是一个张量，表示`ys`中对应元素关于`xs`中对应元素的梯度：若`ys`和`xs`都是长度为`n`的张量列表，那么`tf.gradients(ys,xs)`的输出就是一个长度为`n`的张量列表，其中第`i`个张量是`ys[i]`关于`xs[i]`的梯度；若`ys`是一个张量，`xs`是一个张量列表，那么`tf.gradients(ys, xs)`的输出就是一个长度与`xs`相同的张量列表，其中第`i`个张量是`ys`关于`xs[i]`的梯度。
        u_x=tf.gradients(u,x)[0]  #计算u关于x的梯度
        v_x=tf.gradients(v,x)[0]  #计算v关于x的梯度
        return u,v,u_x,v_x
    


    #定义了一个名为net_f_uv的函数/方法，用于计算f_u和f_v。这个方法接受两个参数，分别是x和t，其中x是输入数据，t是时间数据。最后返回计算得到的f_u和f_v。
    def net_f_uv(self, x, t):
        u,v,u_x,v_x=self.net_uv(x,t) #调用上面的函数/方法，计算神经网络的输出（两个）以及输出关于输入x的梯度（两个）
        u_t=tf.gradients(u,t)[0] #计算u关于t的梯度
        u_xx=tf.gradients(u_x,x)[0] #计算u_x关于x的梯度，也就是u关于x的二阶导数
        v_t=tf.gradients(v,t)[0] #计算v关于t的梯度
        v_xx=tf.gradients(v_x,x)[0] #计算v_x关于x的梯度，也就是v关于x的二阶导数
        
        f_u=u_t+0.5*v_xx+(u**2+v**2)*v    #计算f_u,定义见论文
        f_v=v_t-0.5*u_xx-(u**2+v**2)*u   #计算f_v,定义见论文
        
        return f_u,f_v
    
    def callback(self,loss):  #定义了一个名为callback的函数/方法，打印损失值
        print('Loss:',loss)


    
    #定义了一个名为train的函数/方法，用于训练神经网络。这个方法接受一个参数nIter，表示训练的迭代次数。
    def train(self, nIter):
        #创建一个名为tf_dict的字典，该字典将TensorFlow占位符映射到它们对应的数据。创建tf_dict的目的是为了在运行TensorFlow的计算图时，能够将数据传递给占位符。例如，当运行self.sess.run(self.train_op_Adam, tf_dict)时，tf_dict中的数据就会被传递给对应的占位符，然后在计算图中使用
            #字典语法：dict = {key1: value1, key2: value2, ...}：key1、key2等是字典的键，value1、value2等是对应的值（这里键是占位符，值是对应的数据）。
        tf_dict = {self.x0_tf:self.x0,self.t0_tf:self.t0,
                   self.u0_tf:self.u0,self.v0_tf:self.v0,
                   self.x_lb_tf:self.x_lb,self.t_lb_tf:self.t_lb,
                   self.x_ub_tf:self.x_ub,self.t_ub_tf:self.t_ub,
                   self.x_f_tf:self.x_f,self.t_f_tf:self.t_f}
        #time.time()函数用于获取当前时间并赋值给start_time

        #writer=tf.summary.FileWriter('logs/',sess.graph) #tensorboard可视化


        start_time = time.time()
        for it in range(nIter):  #进行nIter次训练迭代
            #每次迭代时传入输入数据，并执行Adam优化器的训练操作
                #sess.run(fetches, feed_dict=None, options=None, run_metadata=None)：该函数用于执行TensorFlow操作或评估Tensor对象。第一个参数表示需要执行的操作或需要评估的Tensor；第二个参数feed_dict是一个字典，用于将数据传递给占位符；其他参数用于控制会话的细节，通常不需要修改
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print，如果迭代次数是10的倍数，那么就计算并打印出当前的迭代次数、损失值和训练时间
            if it % 10 == 0:
                elapsed = time.time() - start_time  #每十次迭代计算下运行时间
                loss_value = self.sess.run(self.loss, tf_dict)  #每十次迭代计算下损失值
                print('It: %d, Loss: %.3e, Time: %.2f' %   
                      (it, loss_value, elapsed))  #打印迭代次数、损失值和训练时间
                start_time = time.time()  #更新当前时间，为计算下一个10次迭代的运行时间做准备
        #使用优化器来最小化损失函数，第一个参数表示TensorFlow会话，第二个参数表示将数据传递给占位符的字典，fetches表示要获取的结果（这里只获取了损失函数的值），最后一个参数表示每次优化迭代后调用的回调参数，这里是之前定义的，用来打印损失函数   ？？？？                                                                                                              
        #？？？
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss],    
                                loss_callback = self.callback)        
                                    
    #定义了一个名为predict的函数/方法，用于预测神经网络的输出。这个方法接受一个参数X_star，表示输入数据。最后返回预测的两个输出和两个输出的梯度。
    def predict(self, X_star):
        #创建一个字典，其中包含了两个键值对，键是 TensorFlow 的占位符（self.x0_tf 和 self.t0_tf），值是 X_star中的对应列。这个字典将被用于在下面的TensorFlow 会话中给placeholder赋值，用以运行计算图。
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        #使用 self.sess.run方法来给placeholder赋值并运行计算图，并获取 self.u0_pred(神经网络的输出)和self.v0_pred 的计算结果。这两个结果被保存在 u_star 和 v_star 中
        u_star = self.sess.run(self.u0_pred, tf_dict)  
        v_star = self.sess.run(self.v0_pred, tf_dict)  
        #创建了另一个字典tf_dict，其中包含了两个键值对，键是TensorFlow的占位符（self.x_f_tf和self.t_f_tf）值是 X_star中的对应列。这个字典将被用于在 TensorFlow 会话中运行计算图
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        #使用 self.sess.run方法来运行计算图给placeholder赋值并运行计算图，并获取 self.f_u_pred(神经网络的输出)和self.f_v_pred 的计算结果。这两个结果被保存在 f_u_star 和 f_v_star 中
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
               
        return u_star, v_star, f_u_star, f_v_star


#Python中的一个常见模式。在Python文件被直接运行时，特殊变量__name__的值为"__main__"。所以，这行代码的意思是：如果这个文件被直接运行（而不是被导入作为模块），那么就执行后面的代码
if __name__ == "__main__":   #这种模式常常用于在一个Python文件中区分出哪些代码是用于定义函数、类等，哪些代码是用于直接执行的。这样，当这个文件被导入作为模块时，只有函数和类的定义会被执行，而直接执行的代码则不会被执行
    #设置噪声值为0 
    noise = 0.0        
    
    # Doman bounds，定义两个一维数组lb和ub，问题域是一个二维空间，其中 x 的范围是 -5 到 5，t 的范围是 0 到 π/2(竖着的)
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])
    #定义三个整数，分别表示初始条件点数量、边界条件点数量和在问题域内部的点的数量（这些点用于训练神经网络）
    N0 = 50
    N_b = 50
    N_f = 20000
    #定义一个列表layers，其中包含了神经网络的层数和每一层的神经元数量
    layers = [2, 100, 100, 100, 100, 2]
    #读取名为NLS.mat的Matlab文件，文件中的数据存储在data变量中。这里的路径也要随着设备的情况修改    
    data = scipy.io.loadmat('C:/Users/cheny/Documents/GitHub/PINNs/main/Data/NLS.mat')
    #从data字典中取出变量tt和x的值，并转换为一维数组（flatten方法），最后tongg[:,None]将一维数组转换为二维数组
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu'] #从data字典中取出变量uu的值，并赋值给Exact
    Exact_u = np.real(Exact)  #取Exact的实部，赋值给Exact_u
    Exact_v = np.imag(Exact)  #取Exact的虚部，赋值给Exact_v
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2) #计算复数uu的|uu|
    #生成一个二位网络，X和T是输出的二维数组
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))  #X_star是一个二维数组，其中第一列是X的展平，第二列是T的展平
    u_star = Exact_u.T.flatten()[:,None] #先对Exact_u进行转置，然后使用flatten方法将其转换为一维数组，最后使用[:,None]将其转换为二维数组
    v_star = Exact_v.T.flatten()[:,None] #同上，比如Exact_v是m*n二维数组，Exact_v.T是n*m二维数组，Exact_v.T.flatten()是一个长度为n*m的一维数组，Exact_v.T.flatten()[:,None]是一个(n*m)*1的三维数组
    h_star = Exact_h.T.flatten()[:,None]
    #上面五行代码的意义见Numpy库的索引的介绍


    ###########################
    
    #从0~数组x的行数(256)中随机选择N0个数，replace=False表示不允许重复选择，最后将这N0个数赋值给idx_x
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    #从x中选择N0个对应的行(idx_x对应的行)，最后将这N0行赋值给x0
    x0 = x[idx_x,:]
    #从Exact_u中选择N0个对应的行(idx_x对应的行)的第一列元素，最后将这N0个元素赋值给u0
    u0 = Exact_u[idx_x,0:1]
    v0 = Exact_v[idx_x,0:1]
    #从0~数组t的行数中随机选择N_b个数，replace=False表示不允许重复选择，最后将这N_b个数赋值给idx_t
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    #从t中选择N_b个对应的行(idx_t对应的行)，最后将这N_b行赋值给tb
    tb = t[idx_t,:]
    
    X_f = lb + (ub-lb)*lhs(2, N_f) #lhs函数采用拉丁超采样方法，生成一个近似均匀分布的多维样本点集，返回的是一个形状为（$N_f$，2）的数组，每一行都是一个2维的样本点，所有样本点都在[0,1]范围内，并对该样本集进行缩放，把每个样本从[0,1]区间缩放到[lb,ub]区域内，即得到了指定范围内均匀分布的样本$X_f$。

    #创建PINN模型并输入各种参数        
    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)
    #获取当前时间并赋值给start_time          
    start_time = time.time()       
    #训练模型50000次         
    model.train(50000)
    #获取当前时间并减去start_time，得到训练时间并赋值给elapsed
    elapsed = time.time() - start_time                
    #打印训练所需时间
    print('Training time: %.4f' % (elapsed))
    
    #用训练好的模型进行预测，返回四个值（均为数组）    
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    #计算u_pred和v_pred的模（平方和的平方根），赋值给h_pred
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    #计算误差（基于2范数）        
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    #打印误差
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    #使用griddata函数将X_star、u_pred、v_pred和h_pred插值到网格上，得到U_pred、V_pred和H_pred
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
    #同上，使用griddata函数将X_star、f_u_pred和f_v_pred插值到网格上，得到FU_pred和FV_pred
    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')     
    

    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    #调用concatenate函数拼接数组
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])  #(X0;X_lb;X_ub)
    #调用plotting文件中的newfig函数，生成一个宽1英寸、高0.9英寸的图像fig和子图ax
    fig, ax = newfig(1.0, 0.9) #这里ax是一个axes对象，代表子图，figure是一个figure对象，是一个图形窗口，代表整个图形
    ax.axis('off') #关闭子图的轴的显示
    
    ####### Row 0: h(t,x)，绘制第一个子图，展示x,t和|h(t,x)|的关系##################    
    #创建一个包含子图的网格，1行2列
    gs0 = gridspec.GridSpec(1,2)  #创建一个1×2的网络，用于存放子图
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0) #更新该网络的参数，第一个表示子图的顶部位置为0.94，第二个参数表示子图的底部位置为0.667，第三个表示子图左侧的位置为0.15，第四个参数表示子图的右侧位置为0.85，第五个参数表示子图之间的宽度，0表示子图之间没有空隙
    ax = plt.subplot(gs0[:,:]) #在gs0[:,:] 指定的位置创建了一个子图，并将返回的axes对象赋值给ax。gs0[:,:]表示GridSpec对象gs0的所有行和所有列，所以这行代码创建的子图占据了整个图形。
    
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')  #imshow函数用于显示图像，接受一些参数，第一个参数是图像数据，这里是H_pred的转置；第二个参数是插值方法（用于在像素之间插入新的像素），这里是最邻近插值；
                                                  #第三个参数是颜色映射，这里是从黄色Yl到绿色Gn再到蓝色Bu；第四个参数是图像的范围，这里lb和ub分别是数据的下界和上界；第五个参数是图像的原点位置，这里表示原点在右下角；第六个参数是图像的纵横比，这里表示调整横纵比以填充整个axes对象
                                                  #最后的结果返回一个axesimage对象，也就是h，可以通过这个对象进一步设置图像的属性
    divider = make_axes_locatable(ax)  #使用 make_axes_locatable 函数创建了一个 AxesDivider 对象。这个函数接受一个 Axes 对象作为参数，返回一个 AxesDivider 对象。AxesDivider 对象可以用来管理子图的布局，特别是当你需要在一个图的旁边添加另一个图时。
    cax = divider.append_axes("right", size="5%", pad=0.05) #使用append_axes方法在原始轴的右侧添加了一个新的轴。append_axes 方法接受三个参数：位置（"right"）、大小（"5%"）和间距（0.05）。在原始轴的右侧添加了一个新的轴，新轴的大小是原始轴的 5%，新轴与原始轴之间的间距是 0.05 英寸
    fig.colorbar(h, cax=cax) #使用colorbar方法在新轴上添加了一个颜色条。colorbar 方法接受两个参数：axesimage 对象（h）和新轴（cax）。
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False) #在ax上绘制散点图，前两个参数是散点的x坐标和y坐标；kx表示黑色的x（散点形状是x），label是散点的标签，clip_on表示散点可以绘制在轴的边界外
    #在ax图上绘制三条虚线
    line = np.linspace(x.min(), x.max(), 2)[:,None] #生成了一个包含2个等间距的数值的数组，这些数值在 x.min() 到 x.max() 之间。[:,None] 是一个索引操作，用于将一维数组转换为二维数组。这里其实就是[-5;5]
    #第一个参数是虚线的x坐标，line是虚线y的坐标，第三个参数是虚线的样式，k表示黑色，--表示虚线，最后一个参数表示虚线的参数是1
    ax.plot(t[75]*np.ones((2,1)),line,'k--',linewidth=1) 
    ax.plot(t[100]*np.ones((2,1)),line,'k--',linewidth=1)
    ax.plot(t[125]*np.ones((2,1)),line,'k--',linewidth=1)    
    #设置ax子图的x轴的标签为t，y轴的标签为x。这里$t$和$x$是latex格式的文本，用于生成数学公式
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    #设置子图ax的图例，frameon=False表示不显示图例的边框，loc='best'表示图例的位置是最佳位置，最后返回的leg是一个legend对象，表示图形的图例
    leg = ax.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), color='w')   #用来设置图例中文本的颜色，这里是白色，取消注释后文本会变为白色
    ax.set_title('$|h(t,x)|$', fontsize = 10) #设置子图ax的标题为$|h(t,x)|$，表示latex格式的文本，用于生成数学公式，fontsize=10表示字体大小为10
    
    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1,3) #创建一个1×3的网络，用于存放子图
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5) #更新该网络的参数，第一个表示子图的顶部位置为0.667，第二个参数表示子图的底部位置为0，第三个表示子图左侧的位置为0，第四个参数表示子图的右侧位置为0.9，第五个参数表示子图之间的宽度为0.5
    
    ax = plt.subplot(gs1[0,0])  #在gs1[0,0]指定的位置，也就是网格的第一行第一列，创建了一个子图，并将返回的axes对象赋值给ax。
    #绘制了两条线，一条表示精确值，一条表示预测值
    ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')      #第一个参数表示x轴上的坐标；第二个参数表示y轴上的坐标；第三个参数b-表示蓝色的实线；linewidth表示线的宽度为2；label表示线的标签
    ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction') #同上
    #设置ax子图的x轴的标签为x，y轴的标签为|h(t,x)|。这里$x$和$|h(t,x)|$是latex格式的文本，用于生成数学公式
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    #设置子图的标题，几个子图标题随着t的变化而变化，字体大小为10 
    ax.set_title('$t=%.2f$' % (t[75]), fontsize = 10)
    ax.axis('square') #设置子图的纵横比，使得x轴和y轴的单位长度相等，形成一个正方形的区域
    ax.set_xlim([-5.1,5.1]) #第一个子图的x轴范围是-5.1到5.1
    ax.set_ylim([-0.1,5.1]) #第一个子图的y轴范围是-0.1到5.1
    
    ax = plt.subplot(gs1[0, 1]) #在gs1[0,1]指定的位置，也就是网格的第一行第二列，创建了一个子图，并将返回的axes对象赋值给ax。
    #绘制了两条线，一条表示精确值，一条表示预测值
    ax.plot(x,Exact_h[:,100],'b-', linewidth = 2, label = 'Exact')        #第一个参数表示x轴上的坐标；第二个参数表示y轴上的坐标；第三个参数b-表示蓝色的实线；linewidth表示线的宽度为2；label表示线的标签
    ax.plot(x,H_pred[100,:],'r--', linewidth = 2, label = 'Prediction')   #同上
    ax.set_xlabel('$x$') #设置子图的x轴的标签为x
    ax.set_ylabel('$|h(t,x)|$') #设置子图的y轴的标签为|h(t,x)|
    ax.axis('square')   #设置子图的纵横比，使得x轴和y轴的单位长度相等，形成一个正方形的区域
    ax.set_xlim([-5.1,5.1])     #第二个子图的x轴范围是-5.1到5.1
    ax.set_ylim([-0.1,5.1])     #第二个子图的y轴范围是-0.1到5.1
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)        #设置第二个子图的标题，标题随着t的变化而变化，字体大小为10
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)  #设置第二个子图的图例，loc='upper center'表示图例的位置是上方中心，bbox_to_anchor=(0.5,-0.8)表示图例的中心位置是在子图的中间偏下方0.8的位置，ncol=5表示图例的列数是5，frameon=False表示不显示图例的边框
    
    ax = plt.subplot(gs1[0, 2]) #在gs1[0,2]指定的位置，也就是网格的第一行第三列，创建了一个子图，并将返回的axes对象赋值给ax。
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')        #第一个参数表示x轴上的坐标；第二个参数表示y轴上的坐标；第三个参数b-表示蓝色的实线；linewidth表示线的宽度为2；label表示线的标签
    ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')    #同上
    ax.set_xlabel('$x$') #设置子图的x轴的标签为x
    ax.set_ylabel('$|h(t,x)|$') #设置子图的y轴的标签为|h(t,x)|
    ax.axis('square')    #设置子图的纵横比，使得x轴和y轴的单位长度相等，形成一个正方形的区域
    ax.set_xlim([-5.1,5.1])    #第三个子图的x轴范围是-5.1到5.1
    ax.set_ylim([-0.1,5.1])    #第三个子图的y轴范围是-0.1到5.1
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)    #设置第三个子图的标题，标题随着t的变化而变化，字体大小为10
    
    savefig('C:/Users/cheny/Documents/GitHub/PINNs/main/continuous_time_inference (Schrodinger)/figures/NLS')  #用来保存图形，将当前图形保存为名为‘NLS’的文件，保存到位置是当前目录下的‘figures’文件夹；这里的路径也要随着设备的情况修改。注意这边来必须提前建立好figures文件夹，否则会报错
    #在文件路径中，"."和".."有特殊的含义。"."表示当前目录，".."表示上一级目录。例如，如果你在"/home/user/documents"目录下，"."就表示"/home/user/documents"，而".."表示"/home/user"。"当前文件夹"通常指的是正在执行的脚本所在的文件夹。在Python中，你可以使用os.getcwd()来获取当前工作目录。
    print('End')
