## Анализ данных и исследование зависимости доходностей акций разного объема торгов.
### Данные: ГазпромНефть, РосНефть (SIBN, ROSN)


```python
# Импорт всех библиотек
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
import seaborn as sns
import matplotlib
```


```python
def read(x): # Функция импорта исходных данных и их обработка
    return pd.read_csv(x, names=["Date", "Time", "Open", "High", "Low", "Close",
                                           "Volume"], skiprows=1, parse_dates=True, delimiter=",")
```


```python
# Исходные данные, создание столбца
rosn = ["ROSN_Hour.csv", "ROSN_Day.csv", "ROSN_Week.csv", "ROSN_Month.csv"]
sibn = ["SIBN_Hour.csv", "SIBN_Day.csv", "SIBN_Week.csv", "SIBN_Month.csv"]
for i in range(len(rosn)): # Обработка и создание колонки с лог. доходностями
    rosn[i] = read(rosn[i])
    sibn[i] = read(sibn[i])
    rosn[i]["Date"] = pd.to_datetime(rosn[i]["Date"], format='%Y%m%d', errors='ignore')
    sibn[i]["Date"] = pd.to_datetime(sibn[i]["Date"], format='%Y%m%d', errors='ignore')
    rosn[i]["LogProfit"] = (np.log(rosn[i]["Close"]) - np.log(rosn[i]["Open"]))
    sibn[i]["LogProfit"] = (np.log(sibn[i]["Close"]) - np.log(sibn[i]["Open"]))
```


```python
# Создание массива с лог доходностями за кварталы Роснефть
CloseRosn = list(rosn[3]["Close"])
OpenRosn = list(rosn[3]["Open"])
rosnQuarter = list()
for i in range(0, len(CloseRosn[:9]), 3):
    rosnQuarter.append(np.log(CloseRosn[i + 2] / OpenRosn[i]))
rosnQuarter.append(np.log(CloseRosn[-1] / OpenRosn[-1]))

# Создание массива с лог доходностями за кварталы Газпром Нефть
CloseSibn = list(sibn[3]["Close"])
OpenSibn = list(sibn[3]["Open"])
sibnQuarter = list()
for i in range(0, len(CloseSibn[:9]), 3):
    sibnQuarter.append(np.log(CloseSibn[i + 2] / OpenSibn[i]))
sibnQuarter.append(np.log(CloseSibn[-1] / OpenSibn[-1]))
```


```python
# Функции для рассчетов
def datas(List):
    m = int(1 + 3.322*np.log(len(List))) # Вычисляем M интервалов
    step = ((max(List)) - (min(List))) / (m - 1) # Вычисляем шаг (дельта r)
    ranges = [] # Массив интервалов
    for i in range(m):
        ranges.append(min(List) + i * step)
    data = []
    for j, r in enumerate(ranges[:-1]):
        count = 0
        for i in List:
            if i >= r and i < ranges[j + 1]:
                count += 1
        data.append(count) # Массив с количеством значений в каждом из интервалов
    return step, m, data, ranges

def plot(List, ListOfLogFor): # Вычисления по формулам функции плотности вероятностей
    x = [List[3][i] + List[0] / 2 for i in range(len(List[2]))]
    y = [1 / List[0] * (List[2][i] / len(ListOfLogFor)) for i in range(len(List[2]))]
    return x,y
```


```python
# Построение графика функции плотности распределения вероятностей лог доходностей РосНефти (период = час)
coor = plot(datas(rosn[0]["LogProfit"]), rosn[0]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("РосНефть, час")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2e56e048>]




    
![png](images/output_6_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(rosn[0])))

chart = sns.distplot(rosn[0]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_7_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей РосНефти (период = день)
coor = plot(datas(rosn[1]["LogProfit"]), rosn[1]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("РосНефть, день")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2e56e518>]




    
![png](images/output_8_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(rosn[1])))

chart = sns.distplot(rosn[1]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_9_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей РосНефти (период = неделя)
coor = plot(datas(rosn[2]["LogProfit"]), rosn[2]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("РосНефть, неделя")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2e69a7b8>]




    
![png](images/output_10_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(rosn[2])))

chart = sns.distplot(rosn[2]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_11_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей РосНефти (период = месяц)
coor = plot(datas(rosn[3]["LogProfit"]), rosn[3]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("РосНефть, месяц")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2b6df2e8>]




    
![png](images/output_12_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(rosn[3])))

chart = sns.distplot(rosn[3]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_13_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей РосНефти (период = квартал)
coor = plot(datas(rosnQuarter), rosnQuarter)
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("РосНефть, квартал")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2b54bd30>]




    
![png](images/output_14_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(rosnQuarter)))

chart = sns.distplot(rosnQuarter, hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_15_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей ГазпромНефти (период = час)
coor = plot(datas(sibn[0]["LogProfit"]), sibn[0]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("ГазпромНефть, час")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2e5abb38>]




    
![png](images/output_16_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(sibn[0])))

chart = sns.distplot(sibn[0]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_17_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей ГазпромНефти (период = день)
coor = plot(datas(sibn[1]["LogProfit"]), sibn[1]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("ГазпромНефть, день")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2cfba240>]




    
![png](images/output_18_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(sibn[1])))

chart = sns.distplot(sibn[1]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_19_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей ГазпромНефти (период = неделя)
coor = plot(datas(sibn[2]["LogProfit"]), sibn[2]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("ГазпромНефть, неделя")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2e61e128>]




    
![png](images/output_20_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(sibn[2])))

chart = sns.distplot(sibn[2]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_21_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей ГазпромНефти (период = месяц)
coor = plot(datas(sibn[3]["LogProfit"]), sibn[3]["LogProfit"])
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("ГазпромНефть, месяц")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2e6756d8>]




    
![png](images/output_22_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(sibn[3])))

chart = sns.distplot(sibn[3]["LogProfit"], hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_23_1.png)
    



```python
# Построение графика функции плотности распределения вероятностей лог доходностей ГазпромНефти (период = квартал)
coor = plot(datas(sibnQuarter), sibnQuarter)
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xlabel("Лог. доходность")
ax.set_ylabel("Плотность")
ax.set_title("ГазпромНефть, квартал")
ax.plot(coor[0], coor[1])
```




    [<matplotlib.lines.Line2D at 0x21e2e5d3630>]




    
![png](images/output_24_1.png)
    



```python
# Построение графика и гистограммы с помощью Seaborn
m = int(1 + 3.322*np.log(len(sibnQuarter)))

chart = sns.distplot(sibnQuarter, hist=True, kde=True, bins = m,  color = "darkblue", hist_kws = {"edgecolor": "black"}, kde_kws={"linewidth": 3})
```

    C:\Python\Anaconda\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


    
![png](images/output_25_1.png)
    



```python
# Построение матрицы коэффициентов корреляцции РосНефть
rosnQuarter = pd.Series(rosnQuarter)
corrRosn = pd.DataFrame(data = {"hour": rosn[0]["LogProfit"],
                           "day": rosn[1]["LogProfit"], 
                           "week": rosn[2]["LogProfit"],
                           "month": rosn[3]["LogProfit"],
                           "quarter": rosnQuarter})
corrRosn = corrRosn.corr()
print("Матрица коэффициентов корреляции по различным базовым периодам. РосНефть")
corrRosn
```

    Матрица коэффициентов корреляции по различным базовым периодам. РосНефть
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>day</th>
      <th>week</th>
      <th>month</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour</th>
      <td>1.000000</td>
      <td>-0.090273</td>
      <td>-0.032202</td>
      <td>0.327230</td>
      <td>-0.635061</td>
    </tr>
    <tr>
      <th>day</th>
      <td>-0.090273</td>
      <td>1.000000</td>
      <td>-0.174857</td>
      <td>-0.225449</td>
      <td>0.035265</td>
    </tr>
    <tr>
      <th>week</th>
      <td>-0.032202</td>
      <td>-0.174857</td>
      <td>1.000000</td>
      <td>-0.086116</td>
      <td>-0.666238</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.327230</td>
      <td>-0.225449</td>
      <td>-0.086116</td>
      <td>1.000000</td>
      <td>-0.616397</td>
    </tr>
    <tr>
      <th>quarter</th>
      <td>-0.635061</td>
      <td>0.035265</td>
      <td>-0.666238</td>
      <td>-0.616397</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# График, строка 1 (час)
y = list(corrRosn["hour"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("РосНефть, час")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2e838eb8>]




    
![png](images/output_27_1.png)
    



```python
# График, строка 2 (день)
y = list(corrRosn["day"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("РосНефть, день")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2e8b6f28>]




    
![png](images/output_28_1.png)
    



```python
# График, строка 3 (неделя)
y = list(corrRosn["week"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("РосНефть, неделя")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2e8f6fd0>]




    
![png](images/output_29_1.png)
    



```python
# График, строка 4 (месяц)
y = list(corrRosn["month"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("РосНефть, месяц")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2e952b70>]




    
![png](images/output_30_1.png)
    



```python
# График, строка 5 (квартал)
y = list(corrRosn["quarter"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("РосНефть, квартал")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2e9b6eb8>]




    
![png](images/output_31_1.png)
    



```python
# Построение матрицы коэффициентов корреляцции ГазпромНефть
sibnQuarter = pd.Series(sibnQuarter)
corrSibn = pd.DataFrame(data = {"hour": sibn[0]["LogProfit"],
                           "day": sibn[1]["LogProfit"], 
                           "week": sibn[2]["LogProfit"],
                           "month": sibn[3]["LogProfit"],
                           "quarter": sibnQuarter})
corrSibn = corrSibn.corr()
corrSibn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>day</th>
      <th>week</th>
      <th>month</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour</th>
      <td>1.000000</td>
      <td>-0.002195</td>
      <td>-0.246653</td>
      <td>0.007286</td>
      <td>0.059566</td>
    </tr>
    <tr>
      <th>day</th>
      <td>-0.002195</td>
      <td>1.000000</td>
      <td>-0.103822</td>
      <td>-0.217214</td>
      <td>0.287790</td>
    </tr>
    <tr>
      <th>week</th>
      <td>-0.246653</td>
      <td>-0.103822</td>
      <td>1.000000</td>
      <td>-0.011257</td>
      <td>-0.505238</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.007286</td>
      <td>-0.217214</td>
      <td>-0.011257</td>
      <td>1.000000</td>
      <td>0.936797</td>
    </tr>
    <tr>
      <th>quarter</th>
      <td>0.059566</td>
      <td>0.287790</td>
      <td>-0.505238</td>
      <td>0.936797</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# График, строка 1 (час)
y = list(corrSibn["hour"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("ГазпромНефть, час")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2b5a6048>]




    
![png](images/output_33_1.png)
    



```python
# График, строка 2 (день)
y = list(corrSibn["day"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("ГазпромНефть, день")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2ea1ee48>]




    
![png](images/output_34_1.png)
    



```python
# График, строка 3 (неделя)
y = list(corrSibn["week"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("ГазпромНефть, неделя")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2ea5ce48>]




    
![png](images/output_35_1.png)
    



```python
# График, строка 4 (месяц)
y = list(corrSibn["month"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("ГазпромНефть, месяц")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2eabdd30>]




    
![png](images/output_36_1.png)
    



```python
# График, строка 5 (квартал)
y = list(corrSibn["quarter"])
x = range(5)
labels = ["Hour", "Day", "Week", "Month", "Quarter"]
matplotlib.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xticklabels(labels)
ax.set_xticks(x)
ax.set_ylabel("Коэффициент корр.")
ax.set_title("ГазпромНефть, квартал")
ax.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x21e2fae8630>]




    
![png](images/output_37_1.png)
    

