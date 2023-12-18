

#!/usr/bin/env python3      
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: mraissi
"""

import numpy as np
import matplotlib as mpl #`matplotlib` 是一个用于创建图表和其他可视化的 Python 库。它提供了大量的工具，可以帮助你创建各种图表，包括线图、散点图、柱状图、饼图、直方图等等。
##mpl.use('pgf') 
#上面这行代码如果取消注释，mpl.use('pgf') 这行代码的作用是设置matplotlib使用pgf作为后端。"后端"是指用于创建图形的底层库。matplotlib支持多种后端，包括用于在屏幕上显示图形的交互式后端，以及用于生成图像文件（如 PNG、PDF、SVG、PGF 等格式）的非交互式后端。
#pgf 是一个用于生成 PostScript 和 PDF 图形的 TeX 宏包。如果你使用 mpl.use('pgf')，matplotlib 将使用 pgf 来生成图形。这在你需要将 matplotlib 生成的图形嵌入到 LaTeX 文档中时非常有用，因为 pgf 生成的图形可以被 LaTeX 直接处理，而无需转换为其他格式。
#然而，由于这行代码被注释掉了，所以 matplotlib 将使用默认的后端。默认的后端取决于你的环境，可能是一个用于在屏幕上显示图形的交互式后端，也可能是一个用于生成 PNG 图像的非交互式后端。

#定义了一个名为figsize的函数，用于计算图形的大小。接受两个参数：scale和nplots。scale 是一个用于调整图形大小的比例因子，nplots是图形的数量，默认值为1。
#这个函数可以用于设置 matplotlib图形的大小，以便它们在LaTeX文档中看起来更美观。最后返回一个代表fig_size的列表
def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    #首先定义了 fig_width_pt，这是图形宽度的点数（pt）。这个值通常从 LaTeX 文档中获取，使用 \the\textwidth 命令。
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    #定义了 inches_per_pt，这是一个用于将点数转换为英寸的比例因子。这个值是固定的，等于 1.0/72.27。
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    #定义了 golden_mean，这是一个美学比例，等于 (np.sqrt(5.0)-1.0)/2.0。这个比例也被称为黄金分割比例，它被认为是最美的比例。
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    #计算了图形的宽度fig_width（以英寸为单位）。这个宽度等于前面定义的点数宽度fig_width_pt乘以前面定义的点数到英寸的转换因子inches_per_pt，再乘以比例因子 scale（输入参数）。
    fig_height = nplots*fig_width*golden_mean              # height in inches
    #计算了图形的高度fig_height（以英寸为单位）。这个高度等于图形数量nplots（输入参数，默认为1）乘以前面计算得到的图形宽度fig_width，再乘以前面定义的黄金分割比例golden_mean。
    fig_size = [fig_width,fig_height]
    #创建了一个由前面计算的图形宽度和高度构成的列表fig_size，并返回这个列表。
    return fig_size

# #这段代码是在配置 matplotlib 以使用LaTeX输出图形。这是通过创建一个名为pgf_with_latex 的字典来实现的，这个字典包含了一系列的配置项。
# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     #设置了LaTeX的编译系统为pdflatex。如果你使用的是 xetex 或 lualatex，你需要修改这个设置。
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     #设置matplotlib使用LaTeX来写所有的文本。
#     "text.usetex": True,                # use LaTeX to write all text
#     #设置默认的字体族为serif（衬线字体）。
#     "font.family": "serif",
#     #设置各种字体的列表。这些列表为空，意味着使用matplotlib生成图形的Python脚本所在的LaTeX文档中继承字体设置。也就是说，matplotlib将使用LaTeX文档中定义的字体设置来渲染图形中的文本。这样可以确保图形的字体与LaTeX文档的字体保持一致，使得图形与文档整体风格一致。。
#     "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     #这几行在设置 matplotlib 图形的各种字体大小。这些设置是通过修改 matplotlib 的配置参数（rcParams）来实现的。
#     "axes.labelsize": 10,               # LaTeX default is 10pt font.坐标轴标签的字体大小为 10 点。这是 LaTeX 的默认字体大小
#     "font.size": 10,                    # 图形中其他文本的默认字体大小为 10 点
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller,图例的字体大小为 8 点。这是为了让图例的字体稍微小一些，以便不会干扰到图形的主要内容
#     "xtick.labelsize": 8,               #这两行分别设置了 x 轴和 y 轴刻度标签的字体大小为 8 点
#     "ytick.labelsize": 8,            
#     "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth,设置了图形的默认大小为文本宽度的 1.0 倍。这里使用了前面定义的 figsize 函数来计算图形的大小
#     #这段代码是在设置 matplotlib 生成图形时使用的 LaTeX 预设命令。这些命令被放在 "pgf.preamble" 的列表中。这些预设命令将在 matplotlib 生成图形时被使用。这意味着你可以在图形的文本中使用 LaTeX 的功能，例如数学公式、特殊字符等等。
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)导入了inputenc宏包，并设置了其选项为 utf8x。inputenc宏包用于设置LaTeX文档的输入编码，utf8x选项表示使用UTF-8编码。这意味着你可以在图形的文本中直接使用 UTF-8 编码的字符，例如中文、日文、希腊字母等等。
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble,导入了fontenc宏包，并设置了其选项为 T1。fontenc 宏包用于设置LaTeX文档的字体编码，T1选项表示使用T1编码。T1编码支持西欧语言的大多数字符，包括一些特殊的字符，如法语的重音字符等。
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex) #将这些设置应用到 matplotlib。rcParams 是 matplotlib 的配置对象，update 方法可以用来修改这些配置。

# import matplotlib.pyplot as plt #pyplot 是 matplotlib 库的一个子模块，它提供了一种简单的方式来创建图表。pyplot 提供了大量的函数，用于创建各种图表，包括线图、柱状图、散点图、饼图等。这些函数大多数都提供了丰富的参数，可以用来定制图表的各个方面，如颜色、线型、标签等。其设计风格类似于 MATLAB，它的函数通常会改变当前的图或者子图，这使得它非常适合用于交互式环境，如 Jupyter notebook。

# I make my own newfig and savefig functions
#这段代码定义了一个名为newfig的函数，该函数用于创建一个新的图形（figure）和子图（axes）。这个函数接受两个参数：width和nplots。width 用于指定图形的宽度，nplots 用于指定子图的数量，默认为 1。最后，函数返回了创建的图形和子图。这样，用户就可以在这个图形和子图上进行绘图操作。
def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots)) #通过调用 plt.figure()创建了一个新的图形，这个函数有许多可选参数，其中最常用的是figsize用于指定图形的大小。figsize是一个包含两个元素的元组，分别代表图形的宽度和高度，单位是英寸。
                                                     #例如，plt.figure(figsize=(6, 4)) 将创建一个宽度为 6 英寸、高度为 4 英寸的图形。其大小由前面定义的figsize(width, nplots) 函数计算得出。
    ax = fig.add_subplot(111) #然后，通过调用 fig.add_subplot(111) 在这个图形中添加了一个新的子图，并将其赋值给 ax。
                              #add_subplot 是 matplotlib 库中 Figure 类的一个方法，用于在图形中添加一个子图。其参数是一个三位整数，其中第一位代表行数，第二位代表列数，第三位代表子图的索引，索引从1开始。这里111 表示在一个1行1列的网格中的第1个位置创建子图。也就是说，这个图形只有一个子图。
    return fig, ax

#这段代码定义了一个名为 `savefig` 的函数，该函数用于保存当前的matplotlib图形。这个函数接受两个参数：`filename`和`crop`。`filename` 是保存文件的名称（不包含扩展名），`crop` 是一个布尔值，用于指定是否裁剪图形的边缘，默认为 `True`。
def savefig(filename, crop = True):
    if crop == True: #判断是否需要裁剪图形
        #matplotlib.pyplot 的 savefig 函数来保存当前的图形。第一个参数是文件名。在这个例子中，文件名是 '{}.eps'.format(filename)，这是一个 Python 字符串格式化表达式，它会将 filename 的值插入到 {} 的位置，然后添加 .eps 扩展名。所以，如果 filename 的值是 'my_figure'，那么文件名就会是 'my_figure.eps'。
        #bbox_inches参数用于指定保存的图形的边界框。在这个例子中，bbox_inches='tight' 表示将边界框设置为紧密模式，这会自动裁剪掉图形周围的空白部分。
        #pad_inches 参数用于指定边界框与图形之间的间距。在这个例子中，pad_inches=0 表示边界框紧贴图形，没有额外的间距。

#       plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)#使用`plt.savefig`函数保存图形，并设置`bbox_inches='tight'` 和 `pad_inches=0`，这两个参数会使 matplotlib 自动裁剪图形的空白边缘。图形将被保存为 PDF 格式
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)#同上，图形将被保存为 EPS 格式
    else:     #如果图形不需要裁剪
#       plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))#直接保存图形，不进行裁剪。图形同样将被保存为 PDF 格式和 EPS 格式。
        plt.savefig('{}.eps'.format(filename))


# # Simple plot,一段简单的示例，用于绘制指数移动平均（EMA）图
# fig, ax  = newfig(1.0)

# def ema(y, a):  #定义了一个名为ema的函数用于计算指数移动平均。接受两个参数：y是一个包含数据的列表，a是平滑因子。
#    s = []
#    s.append(y[0])  #函数首先将y的第一个元素添加到结果列表s中
#    for t in range(1, len(y)):
#        s.append(a*y[t]+(1-a)*s[t-1]) #对y的剩余元素进行迭代，每次迭代都计算，并将结果添加到s中
#    return np.array(s) #最后，函数返回s的numpy数组形式
   
# y = [0]*200 #创建了一个名为y的列表，该列表前200个元素为0
# #这行代码使用了列表的`extend`方法。用于将一个列表（或任何可迭代对象）的所有元素添加到当前列表的末尾。
# #这里`[20]*(1000-len(y))` 创建了一个新的列表，这个列表的长度为 `1000-len(y)`，所有元素都是 `20`。然后，这个新列表的所有元素都被添加到 `y` 的末尾
# y.extend([20]*(1000-len(y)))  #将y的后800个元素设置为20
# s = ema(y, 0.01) #使用上面定义的ema函数来计算y的指数移动平均，平滑因子为0.01

# #下面的代码中，`ax` 是一个 `AxesSubplot` 对象，之前它是通过调用`fig.add_subplot(111)` 创建的。`AxesSubplot` 对象代表了图形中的一个子图。
# ax.plot(s) # ax.plot(s) 在子图ax上绘制一个线图，线图的数据来源于's'。然后，使用 ax.set_xlabel('X Label') 和 ax.set_ylabel('EMA') 设置了 x 轴和 y 轴的标签
# ax.set_xlabel('X Label') #设置子图ax的x轴标签为'X Label'
# ax.set_ylabel('EMA') #设置子图ax的y轴标签为'EMA'

# savefig('ema') #将图形保存为文件，文件名为'ema'，默认情况下，文件会被保存在当前的工作目录
