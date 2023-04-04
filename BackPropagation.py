import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

class BpNeural:
    '''
    使用误差逆传播算法实现的单隐藏层神经网络的训练
    '''
    def __init__(self,l=0,q=0,d=0,epoches=100,lr=0.01):
        '''
        神经网络的参数初始化
        :param l: 输出神经元的个数(输出向量的维度)
        :param q: 隐藏层神经元的个数(隐藏层输出向量的维度)
        :param d: 输入层神经元的个数(输入向量的维度)
        :param epoches:训练的轮数
        :param lr:学习率
        '''
        self.l = l
        self.q = q
        self.d = d
        self.epoches = epoches
        self.lr = lr
        #输出神经元的阈值向量
        self.theta = np.random.randn(l)

        #隐藏层到连接层的权值矩阵，其中的第j列表示: 第j个输出神经元被隐藏层所有神经元链接的链接权值向量
        self.W = self._xavier_init(q,l)

        #隐藏层神经元的阈值向量
        self.yt = np.random.randn(q)

        #输入层到隐藏层的链接权值矩阵，其中的第h列表示: 第h个隐藏层神经元被所有输入层神经元链接的权值向量
        self.V = self._xavier_init(d,q)

    def _xavier_init(self,n_inputs, n_outputs):
        """
        使用Xavier初始化权重矩阵
        """
        # 计算Xavier初始化时的标准差
        std_dev = np.sqrt(2.0 / (n_inputs + n_outputs))
        # 生成随机矩阵
        weights = np.random.randn(n_inputs, n_outputs) * std_dev
        return weights

    @classmethod
    def sigmoid(cls,X):
        return 1.0/(1+np.exp(-X))

    def _predict(self,x):
        '''
        输入向量计算其预测向量的值
        :param x: x是一个向量(请最好传入一维的numpy.array对象)
        :return: b,y_estimate b:隐藏层输出,y_estimate: 输出层输出
        '''
        b = np.zeros(self.q)
        for h in range(self.q):
            b[h] = np.dot(x,self.V[:,h])

        b = self.sigmoid(b-self.yt)

        y_estimate = np.zeros(self.l)
        for j in range(self.l):
            y_estimate[j] = np.dot(b,self.W[:,j])
        y_estimate =  self.sigmoid(y_estimate-self.theta)

        return b,y_estimate

    def training(self,X,Y):
        '''
        Bp算法训练神经网络,请务必保证X的样本数目和Y的样本数目相等
        :param X:训练输入矩阵
        :param Y: 训练输出矩阵
        '''

        for _ in range(self.epoches):
            #训练设定好的轮数
            for a in range(X.shape[0]):
                #取出对应的输入向量和输出向量
                x,y = X[a],Y[a]
                if not isinstance(y,np.ndarray):
                    tmp = np.array([0])
                    tmp[0] = y
                    y = tmp
                #计算当前神经网络的隐藏层输出向量和预测输出向量
                b,y_estimate = self._predict(x)

                #初始化梯度算子向量并计算
                g = np.zeros(self.l)
                for j in range(self.l):
                    g[j] = y_estimate[j]*(1-y_estimate[j])*(y[j]-y_estimate[j])

                #初始化e算子向量并计算之
                e = np.zeros(self.q)
                for h in range(self.q):
                    e[h] = b[h]*(1-b[h])*np.dot(g,self.W[h,:])

                #梯度下降更新神经网络参数theta向量
                self.theta =self.theta - self.lr*g

                #梯度下降更新神经网络参数W矩阵
                for j in range(self.l):
                    #对于W的每一列
                    for h in range(self.q):
                        self.W[h][j]=self.W[h][j]+self.lr*b[h]*g[j]

                #梯度下降更新神经网络参数yt向量
                self.yt = self.yt - self.lr * e

                #梯度下降更新神经网络参数V矩阵
                for h in range(self.q):
                    #对于V的每一列
                    for i in range(self.d):
                        self.V[i][h] = self.V[i][h] + self.lr * e[h] * x[i]

    def batch_predict(self,X):
        '''
        批量预测输入数据的输出
        :param X: 输入的批量数据矩阵
        :return:预测值矩阵或者向量
        '''
        y = np.zeros([X.shape[0],self.l])
        for i in range(X.shape[0]):
            _,y[i] = self._predict(X[i])
            y[i] = np.round(y[i])

        return y

def test():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 使用OneHotEncoder进行独热编码
    encoder = OneHotEncoder(categories="auto", sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    k = 5
    # 初始化k折交叉验证
    kf = KFold(n_splits=k, shuffle=True)
    accuracies = []
    nets = []
    #设定一下numpy随机数种子固定一下运算结果
    np.random.seed(12)


    # 循环训练k次，每次使用1/k的数据作为验证集，其余作为训练集
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = y[train_index], y[val_index]

        net = BpNeural(d=4,q=10,l=3,lr=0.01,epoches=1000)
        net.training(X_train,Y_train)
        Y_pred = net.batch_predict(X_val)
        cnt = 0
        for i in range(X_val.shape[0]):
            if (Y_pred[i]==Y_val[i]).all():
                cnt+=1

        nets.append(net)
        accuracies.append(float(cnt/X_val.shape[0]))

    print("5折交叉验证的准确率是",accuracies)
    best_net = nets[accuracies.index(max(accuracies))]
    print("对应的最优的神经网络参数如下: ")
    print("输出神经元的阈值向量为：",best_net.theta)
    print("隐藏神经元与输出神经元的链接权矩阵为：：", best_net.W)
    print("隐藏层神经元的阈值向量为：", best_net.yt)
    print("输入神经元与隐藏层神经元之间的连接权矩阵为：", best_net.V)

test()