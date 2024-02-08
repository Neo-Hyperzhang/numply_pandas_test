import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pandas_datareader.data as web

# obj = pd.Series([4, 7, 1, 2])     #Series的数据结构与表示方法，与Numpy相似（相同数据类型），但具有索引值
# print(obj)
# print(obj.values)
# print(obj.index)

# obj = pd.Series([4, 7, -5, -3], index=['a', 'b', 'c', 'd'])   #为其数据结构选择索引
# print(obj)
# # print(obj[['a', 'b']])   #通过索引选择其值
# # print(obj[obj > 2])      #对其中的数据进行处理，并且保留其索引值
# # print(obj*2)
# # print(np.exp(obj))
# print(bool('b' in obj))    #对索引值进行布尔判定

# dic = {'ohio': 35000, 'Texas': 71000, 'Oregon': 16000}  #通过pandas直接将字典转为Series
# obj = pd.Series(dic)
# print(obj)
# print(type(obj))
# states = ['California', 'ohio', 'Japan']
# obj_re = pd.Series(dic, index=states)   #通过列表按照想要的顺序构造Series，当dic没有所需的键时，其对应的值为NaN
# print(obj_re)
# print(pd.isnull(obj_re))   #通过isnull/notnull判断是否缺失数据
# print(pd.notnull(obj_re))
# obj.name = 'population'      #赋予数据与索引名字
# obj.index.name = 'state'
# print(obj

# obj = pd.Series([4, 7, 1, 2])
# print(obj)
# obj.index = ['a', 'b', 'v', 'e']    #索引值按位置赋值进行改变
# print(obj)


# data = {'state': ['ohio', 'ohio', 'ohio', 'nevada', 'nevada', 'nevada'],      #通过字典形成Dataframe，具有行索引与列索引
#         'year': [2000, 2001, 2002, 2003, 2004, 2022],             #每一列可以是不同的数据类型
#         'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
# # # frame = pd.DataFrame(data)
# # # print(frame)
# # # print(frame.head())   #只选出前五行
# frame = pd.DataFrame(data, columns=['year', 'pop', 'state', 'debt'],
#                      index=['one', 'two', 'three', 'four', 'five', 'six'])  #根据指定顺序排列,改变其列索引
# # print(frame)
# # # print(frame['state'])    #选取某列，检索值为Series
# # # print(frame.year)
# # # print(frame.loc[3])    #选取某行
# # frame['debt'] = 15     #对没有数据的列进行赋值
# # print(frame)
# # frame['debt'] = np.arange(6)
# # print(frame)
# val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])   #创建一个Series并将其赋给DataFrame
# frame['debt'] = val
# print(frame)      #被赋值的列不存在则生成一个新的列
# del frame['debt']
# print(frame)      #del函数删除某行

# pop = {'nevada': {2001: 2.4, 2002: 2.9},
#        'ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
# frame = pd.DataFrame(pop)     #对于嵌套词典
# # print(frame)
# # print(frame.T)
# # print(pd.DataFrame(pop, index=[2001, 2002, 2003]))    #显式的指明索引，内部字典不会被排序
# frame.index.name = 'year'      #赋予name属性
# frame.columns.name = 'state'
# print(frame)
# print(frame.values)     #value属性以ndarry形式返回
# print(type(frame.values))

# obj = pd.Series(range(3), index=['a', 'b', 'c'])     #索引对象，所使用的任意数组或标签序列都可以在内部转换为索引对象
# index = obj.index
# print(index)
# print(obj)
# # index[1]= 'd'   #索引无法被修改
# labels = pd.Index(np.arange(3))
# print(labels)
# obj2 = pd.Series([1.5, 1, 0], index=labels)
# print(obj2)

# obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
# print(obj)
# print(obj.reindex(['a', 'b', 'c', 'd', 'e']))   #重新指定索引——reindex函数用法
# frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
#                      index=['a', 'c', 'd'],
#                      columns=['ohio', 'texas', 'california'])
# print(frame)

# obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
# new_obj = obj.drop('c')    #drop函数通过列值删除行（Series数列）
# print(obj)
# print(new_obj)
# print(obj.drop(['c', 'd']))

# data = pd.DataFrame(np.arange(16).reshape((4, 4)),
#                     index=['ohio', 'colorado', 'utah', 'new york'],
#                     columns=['one', 'two', 'three', 'four'])
# print(data)
# print(data.drop(['colorado', 'ohio']))  #drop函数通过列值删除行（DataFrame数列，drop函数直接操作原对象而不返回新对象

# obj = pd.Series(np.arange(4. ), index=['a', 'b', 'c', 'd'])    #Series数据通过索引值进行选择
# print(obj)
# print(obj['a'])
# print(obj[['a', 'b']])
# print(obj['b':'c'])    #Series的切片包括尾部

# data = pd.DataFrame(np.arange(16).reshape((4, 4)),            #通过索引、切片选择Dataframe的数据
#                     index=['ohio', 'colorado', 'utah', 'new york'],
#                     columns=['one', 'two', 'three', 'four'])
# print(data)
# print(data['two'])
# print(data[['three', 'one']])
# print(data[:2])    #前两行选取

# print(data[data['three'] > 5])     #通过对数据进行判断条件后输出
# data[data < 5] = 0        #通过对数据判断后进行赋值
# print(data)

# print(data.loc['colorado', ['two', 'three']])    #使用loc/iloc函数（[行标签：, [列标签]]）以Numpy风格的语法选取数据
# print(data.iloc[2, [3, 0, 1]])

# s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
# s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
#                index=['a', 'c', 'e', 'f', 'g'])
# print(s1)
# print(s2)
# print(s1 + s2)    #在Series数据中，没有交叠的标签位置上，内部数据会产生缺失值

# df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
#                    index=['ohio', 'texas', 'colorado'])
# df2 = pd.DataFrame(np.arange(12).reshape((4, 3)), columns=list('bde'),
#                    index=['utah', 'ohio', 'texas', 'oregon'])
# print(df1)
# print(df2)
# print(df1 + df2)   #在Series数据中，没有交叠的标签位置上，内部数据会产生缺失值

# df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
#                    columns=list('abcd'))
# df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
#                    columns=list('abcde'))
# df2.loc[1, 'b'] = np.nan  #将第1行b列的值归为缺失
# print(df1)
# print(df2)
# print(df1+df2)
# print(df1.add(df2, fill_value=0))   #有交叠的想加，没交叠的为并集（原数据呈现）

# arr = np.arange(12.).reshape((3, 4))
# print(arr)
# print(arr[0])
# print(arr - arr[0])   #在Numply中每一行都进行相减—广播机制

# frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
#                      columns=list('bde'),
#                      index=['utah', 'ohio', 'texas', 'oregon'])
# series = frame.iloc[0]   #第1行的值与其列索引
# print(frame)
# print(series)
# print(frame - series)    #相减的值自动形成匹配

# series2 = pd.Series(range(3), index=['b', 'e', 'f'])
# print(series2)
# print(frame)
# print(frame + series2)     #索引无交集的情况下，重建索引且合并，其值为缺失

# series3 = frame['d']
# print(frame.sub(series3, axis='index'))  #传递参数axis用来匹配轴，进行广播计算

# frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
#                      index=['utah', 'ohio', 'texas', 'oregon'])
# print(frame)
# # print(np.abs(frame))   #对所有元素取绝对值
# f = lambda x: x.max() - x.min()
# print(frame.apply(f))   #求取每一列中最小值和最大值的差
# print(frame.apply(f, axis='columns'))    #选择参数为行，则为每一行中最小值和最大值的差

# def f(x):
#     return pd.Series([x.min(), x.max()], index=['min', 'max'])
#
# print(frame.apply(f))   #返回带有多个值的Series，也就是Dataframe

# format = lambda x: '%.2f' % x
# print(frame.applymap(format))   #使用applymap对逐个元素进行应用
# print(frame['e'].map(format))   #只对选择的一列使用

# obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
# print(obj)
# print(obj.sort_index())   #根据索引进行排序

# frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
#                      index=['three', 'one'],
#                      columns=['d', 'a', 'b', 'c'])
# print(frame)
# print(frame.sort_index())   #根据行索引进行排序
# print(frame.sort_index(axis=1))   #根据列索引进行排序
# print(frame.sort_index(axis=1, ascending=False))   #根据列索引进行排序，降序排序

# obj = pd.Series([4, 7, -3, 2])
# print(obj)
# print(obj.sort_values())  #根据值进行排序，缺失值排在底部

# frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
# print(frame)
# print(frame.sort_values(by='b'))   #只对列进行排序
# print(frame.sort_values(by=['a', 'b']))   #选定列索引，对指定的列索引进行排序

# obj = pd.Series([7, -5, 7, 4])
# print(obj)
# print(obj.rank())  #输出每一个元素的排名, 默认相同排名取平均值

# frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
#                       'c': [-2, 5, 8, -2.5]})
# print(frame)
# print(frame.rank(axis='columns'))    #对每一行的元素进行排名

# obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
# print(obj)
# print(obj.index.is_unique)   #重复索引的判断
# df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
# print(df)
# print(df.loc['b'])    #在重复索引的情况下，取值


# df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
#                   [np.nan, np.nan], [0.75, -1.3]],
#                   index=['a', 'b', 'c', 'd'],
#                   columns=['one', 'two'])
# print(df)
# # print(df.sum())  #默认为列的和，自动排除NAN（缺失的值）
# # print(df.sum(axis=1)) #行的和
# print(df.idxmax())  #返回某列最大值的索引值
# print(df.cumsum())  #积累型方法，列的叠加和


# data = web.get_data_alphavantage('AAPL', api_key='0RVQJ8CUN9ZFY9ZV')     #测试数据列标签是什么
# print(data)
# all_data = {ticker: web.get_data_alphavantage(ticker, api_key='0RVQJ8CUN9ZFY9ZV')   #调用网站中的数据
#             for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}
# price = pd.DataFrame({ticker: data['close']
#                       for ticker, data in all_data.items()})
# volume = pd.DataFrame({ticker: data['volume']
#                        for ticker, data in all_data.items()})
# returns = price.pct_change()    #调用 pct_change() 方法会计算每列的变化率，即每个股票价格相对于前一个时间点的价格的百分比变化
# # print(price)
# # print(volume)
# print(returns.tail())
# print(returns['MSFT'].corr(returns['IBM']))   #算两只股票的相关性
# print(returns['MSFT'].cov(returns['IBM']))    #算两只股票的协方差
# print(returns.corr())    #以Dataframe的形式返回相关性与协方差矩阵
# print(returns.cov())
# print(returns.corrwith(returns.IBM))    #计算某一列/某个数据的相关性
# print(returns.corrwith(volume))

# obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
# # uniques = obj.unique()   #给出数据中的唯一值
# # print(uniques)
# # print(obj.value_counts())    #给出Series包含的值的个数
# print(pd.value_counts(obj.values, sort=False))    #value_counts是pandas顶层方法，可用于任意数组或序列

# to_much = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
# unique_vals = pd.Series(['c', 'b', 'a'])
# print(pd.Index(unique_vals).get_indexer(to_much))    #针对非唯一数组，利用唯一数组进行分析重复的数组

data = pd.DataFrame({'qu1': [1, 3, 4, 3, 4],
                     'qu2': [2, 3, 1, 2, 3],
                     'qu3': [1, 5, 2, 4, 4]})
print(data)
result =data.apply(pd.value_counts).fillna(0)    #对DataFrame中的每一列进行值计数，并将缺失值填充为 0
print(result)