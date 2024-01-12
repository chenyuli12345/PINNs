%绘制usol的代码

% 获取数组的列数
numColumns = size(usol, 2);
% 获取数组的行数，即每列的数据点数
numRows = size(usol, 1);
% 创建一个x轴的值，从0到列数-1
x = 0:numColumns-1;
% 循环遍历每一行
for i = 1:numRows
    % 绘制图像，纵坐标是usol每列数据情况
    plot(x, usol(i,:));
    % 添加标题
    title(['第', num2str(i), '行数据']);
    % 保存图像，文件名为"第i行数据.png"
    saveas(gcf,['C:/Users/User/Desktop/code/usol数据情况/每行数据（共256行，每行100个数据）/第',num2str(i),'行数据.png']);
end
