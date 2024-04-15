
%修改开头的load文件名
%运行后New_4是对数下不确定度分布情况，从中选取不同不确定度的点
clc,clear
load 'C:\Users\lcy\Documents\GitHub\PINNs\main\continuous_time_inference (Schrodinger)\putongxuedinge.mat'
%% 绘制uncertainty云图
gamma = double(gamma); %转换为双精度浮点数
% X_ing = double(X_xing);
% X = double(X);
X_XING = X_xing;
New =[]; %创建一个空数组
n = size(X,1); %n为X的行数
Cov_X_X =  zeros(n); %n*n的零矩阵

%通过循环，更新Cov_X_X，每个元素是两个向量（X的某两行，X代表神经网络在训练点预测下的隐藏层输出）的平方的负指数
for i =1: n  
    for j =1:n
        xi = X(i,:); %X的第i行的值
        xj = X(j,:); %X的第j行的值
        dij =norm(xi-xj); %计算欧氏距离
        dij = dij^2; 
        Cov_X_X(i,j) =  exp(-gamma*dij);
    end
end
t1 = inv(Cov_X_X); % 第一种求逆
t2 =t1*Cov_X_X; %计算逆矩阵与原矩阵相乘
xish =100;
[u,s,v]=svd(Cov_X_X*xish); %对目标矩阵乘上xish后进行奇异值分解
t3 = xish * inv(v')*inv(s)*inv(u); % 第二种求逆
t4 = t3*Cov_X_X; %计算第二种逆矩阵与原矩阵相乘
differ = t1-t3; %计算差距
New=[];
for loc = 0:1:200  % 对时间t进行循环
    loc
    index= loc*256+(1:256);
    new_1=[];
    for loc_xt=index(1):index(end)
        Cov_xing_xing = 1.0;
        X_xing = X_XING(loc_xt,:);  
        Cov_xing_X=zeros(1,n);
        for j=1:n
            xi = X_xing;
            xj = X(j,:);
            dij =norm(xi-xj);
            dij = dij^2;
            Cov_xing_X(1,j) =  exp(-gamma*dij);
        end
        Cov_X_xing = Cov_xing_X';
        new = Cov_xing_xing-Cov_xing_X*t1*Cov_X_xing;
        new_1=[new_1;new];
    end
    New=[New,new_1];
end
New_1=New;
New_2 = flipud(New_1);

x=(1:201)/200 *pi/2;
y=linspace(-5,5,256);
 y=fliplr(y);
 colormap(jet)
imagesc(x,y,New_2(1:256,1:201));
 colorbar;
set(gca,'YDir','normal')

figure;

% 取对数作图
for i=1:size(New_2,1)
    for j=1:size(New_2,2)
    if New_2(i,j)<0
        New_3(i,j) = 1e-10;
    else
        New_3(i,j) =New_2(i,j);
    end

    end
end
New_4 = log10(New_3);
x=(1:201)/200 *pi/2;
y=linspace(-5,5,256);
y=fliplr(y);
imagesc(x,y,New_4(1:256,1:201));
 colorbar;
set(gca,'YDir','normal')

%save('C:\Users\lcy\Documents\GitHub\Example3-Schrodinger\main_Example3\continuous_time_inference (Schrodinger)\jiuji\buquedingshuju.mat', 'New');