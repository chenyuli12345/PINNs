% 绘制训练点的情况，load文件名修改即可
%% 绘制训练点空间时间分布_1
clear all
load 'C:\Users\lcy\Documents\GitHub\PINNs\main\continuous_time_inference (Schrodinger)\putongxuedinge.mat'

%将所有训练点分为三部分，均为a*2矩阵，第一列表示x，第二列表示t
initial_points =X_know(1:50,:); %前50行代表初值点
bound_points = X_know(51:150,:); %51-150行代表边界值点
collo_points =  X_know(151:end,:); %配位点

hold on %用于在同一图上绘制多个图形
%绘制散点图，第一个表示横坐标（为t），第二个参数表示纵坐标（为x），第三个表示点的样子，o表示圆的点，k表示黑色
plot(initial_points(:,2),initial_points(:,1),'k.') %黑色为初值点
plot(bound_points(:,2),bound_points(:,1),'b.') %蓝色为边界值点
plot(collo_points(:,2),collo_points(:,1),'r.') %红色为配位点
axis([0 pi/2 -5.5 6.5]) %设置坐标轴范围
legend('Initial points','Bound points','Collocation points','Orientation','Horizontal') %添加图例
xlabel('t','FontName','Times New Roman','FontAngle','italic','Fontsize',20) %坐标轴标签
ylabel('x','FontName','Times New Roman','FontAngle','italic','Fontsize',20)

title('所有训练点')