#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Милицына Полина BAE'25
#География литературы: сравнения стран
#Как можно сравнить страны с точки зрения развитости литературной сферы? Рассмотрим 3 критерия: 

#1) количество граждан, получивших литературные премии;
#2) уровень читательской грамотности младших школьников;
#3) анализ пользователей сайта Goodreads на основе данных о стране проживания.

#Так, рассмотрев разные группы населения стран, появится возможность составить комплексное представление об их относительном "литературном климате".


# In[547]:


#Установим часть необходимых библиотек:
get_ipython().system('pip install ipyleaflet')
from ipyleaflet import Map, GeoData, basemaps, LayersControl
get_ipython().system('pip install geopandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install shapely')
get_ipython().system('pip install json')
import geopandas
import json
import shapely
from matplotlib import pyplot as plt
get_ipython().system('pip install selenium')
get_ipython().system('pip install requests')
get_ipython().system('pip install folium matplotlib mapclassify')


# In[ ]:


#Часть 1
#Создадим карту Нобелевской премии по литературе со странами, в которых родились лауреаты Нобелевской премии, а также обозначим на карте их количество.


# In[548]:


#Однако прежде всего получим общую информацию о Нобелевской премии: узнаем о её истории, процедуре отбора кандидатов и критике. Для этого воспользуемся Selenium.
from IPython.display import display
from ipywidgets import Dropdown
import matplotlib.pyplot as plt
from selenium import webdriver

driver=webdriver.Chrome()
driver.get("https://ru.wikipedia.org/wiki/%D0%9D%D0%BE%D0%B1%D0%B5%D0%BB%D0%B5%D0%B2%D1%81%D0%BA%D0%B0%D1%8F_%D0%BF%D1%80%D0%B5%D0%BC%D0%B8%D1%8F_%D0%BF%D0%BE_%D0%BB%D0%B8%D1%82%D0%B5%D1%80%D0%B0%D1%82%D1%83%D1%80%D0%B5")

extra_information=['История','Отбор кандидатов', 'Критика']

d2={}

d2["История"]=driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/p[3]").text
d2["Отбор кандидатов"]=driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/p[5]").text + '\n' + '\n' + driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/ol/li[1]").text + '\n' + driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/ol/li[2]").text + '\n' + driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/ol/li[3]").text + '\n' + driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/ol/li[4]").text + '\n' + '\n' + driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/p[6]").text
d2["Критика"]=driver.find_element("xpath", "/html/body/div[3]/div[3]/div[5]/div[1]/p[7]").text

dropdown = Dropdown(options=[(information, d2[information]) for information in extra_information], description='Доп. инфо.:')

def func(b):
    print(b['new'])

dropdown.observe(func, names='value')
display(dropdown)


# In[549]:


#Теперь перейдем непосредственно к датасету. Для получения необходимых данных воспользуемся API с официального сайта премии. 
#Дисклеймер: дальнейшее присоединение таблицы с бОльшим количеством данных необходимо, так как другие датасеты, доступные через API, 
#были слишком объёмными для цели данного проекта, поэтому в итоге использовалось 2 источника данных.
import requests
import pandas as pd
import re
response1 = requests.get('https://api.nobelprize.org/v1/prize.json?category=literature')
nob = json.loads(response1.text)
#response1.status_code = 200 => соединение установлено
years = []
years_double = []
firstnames = []
surnames = []
for i in nob['prizes']: #nob['prizes'] - список, i - словарь с лауреатом за определенный год
    for j in i.keys(): #j - ключи "year", "category", "laureates", "overallMotivation"
            if j == "overallMotivation":
                firstnames.append("No")
                surnames.append("laureates")
            if j == 'year':
                years.append(int(i[j]))
            if j == "laureates":
               for k in i[j]: #k - словарь как элемент списка(из 1-2 элементов, в зависимости от количества лауреатов за этот год) в значении "laureates" 
                    for d in k: #d - ключи словаря
                        if d == "firstname":
                            firstnames.append(k[d])
                        if d == "surname":
                            surnames.append(k[d])
             
            if j == "laureates": 
                if len(i[j]) > 1: 
                    years.append(int(i['year'])) #если в году было 2 победителя, то год продублируется
                    years_double.append(int(i['year'])) # получим список с годами, в которых было 2 победителя

df_nob = pd.DataFrame({"Year": years, "Firstname": firstnames, "Surname": surnames})

df_nob['Laureate'] = df_nob['Firstname'] + ' ' + df_nob['Surname']
Lastnames=[]
for i in df_nob['Laureate']:
    Lastnames.append(i.split()[-1])

df_nob['Lastname']=Lastnames

df_nob1=df_nob.loc[df_nob['Laureate']!='No laureates']

#Дополним датасет данными о странах рождения, языках написания произведений и возрастом получения премии: 
nob_extra = pd.read_csv('Nobel with countries - Sheet3 (3).csv', delimiter=',', encoding='UTF-8')
surnamesextra=[]
for i in nob_extra['Laureate']:
    surnamesextra.append(i.split()[-1])

nob_extra['Surname']=surnamesextra

#Преобразуем данные:

nob_extra['Age Awarded'] = nob_extra['Age Awarded'].fillna(0)
nob_extra = nob_extra.astype({'Age Awarded': int})
nob_extra1=nob_extra.loc[nob_extra['Laureate']!='No laureates']             

#Перед склеиванием таблиц, на всякий случай, уберем пробелы/переходы на следующую строку в начале и в конце каждой строки:
for i in df_nob1['Laureate']:
    i.strip()
for j in nob_extra['Laureate']:
    j.strip()

#Проверим, что нет однофамильцев и/или людей, получивших премию дважды:
#duplicateRows = nob_extra[nob_extra.duplicated(['Surname'])]
#duplicateRows
#Так как в датафрейме есть 2 лауреата с фамилией Мистраль, изменим фамилию одного из лауреатов и соединим таблицы по столбцу с фамилиями.
nob_extra1.loc[nob_extra1['Laureate'] == 'Gabriela Mistral', 'Surname'] = 'Mistral*'
df_nob1.loc[df_nob1['Laureate'] == 'Gabriela Mistral', 'Lastname'] = 'Mistral*'

nobel = pd.merge(df_nob1, nob_extra1,left_on = 'Lastname', right_on = 'Surname')
yearss=[]
years_x=[]
for year in nobel['Year_x']:
    years_x.append(year)
for year in nob_extra1['Year']:
    if year not in years_x:
        yearss.append(year)
#print(yearss) выводит [1961, 1980, 1994, 2000] - это года, по которым таблицы не соединились. По этим годам таблицы не ссоединились по двум причинам - 1) На сайте Нобеля перепутаны местами имя и фамилия автора (такое наблюдается у Gao Xingjian)
# 2) Потому что некоторые фамилии на википедии и на сайте Нобелевской премии написаны по-разному (к примеру Milosz и Miłosz). Поэтому лауреатов этих 4-х годов добавлю вручную:

nobel.loc[len(nobel.index)] = [2000,'Gao','Xingjian','Gao Xingjian','Xingjian', 2000, 'Gao Xingjian', 'China', 'Chinese', 60, 'Xingjian']
nobel.loc[len(nobel.index)] = [1994,'Kenzaburō','Ōe','Kenzaburō Ōe','Ōe', 1994,'Kenzaburō Ōe', 'Japan', 'Japanese', 59, 'Ōe']
nobel.loc[len(nobel.index)] = [1980,'Czesław','Miłosz','Czesław Miłosz','Miłosz', 1980, 'Czesław Miłosz', 'Poland', 'Polish', 69, 'Miłosz']
nobel.loc[len(nobel.index)] = [1961,'Ivo','Andrić','Ivo Andrić','Andrić', 1961, 'Ivo Andrić', 'Austria-Hungary', 'Serbo-Croatian', 69, 'Andrić']

#Выгрузим датасет со странами:
world = geopandas.read_file('ne_110m_admin_0_countries.zip')
world_new = world[['SOVEREIGNT','SOV_A3','geometry']]
world_new = world_new.rename(columns={'SOVEREIGNT':"Winning Country"})
#Переименуем столбцы для дальнейшего соединения таблиц:
nobel = nobel.rename(columns={'Country':"Winning Country"})
nobel["Winning Country"] = nobel["Winning Country"].apply(lambda x: x.strip())
world_new["Winning Country"] = world_new["Winning Country"].apply(lambda x: x.strip())
#Соединим датасеты:
noble_map = pd.merge(world_new, nobel, left_on = 'Winning Country', right_on = 'Winning Country')
nobel[nobel['Lastname']== 'White']
world_new[world_new['Winning Country'] == 'United Kingdom']
nobel_counted = {}
for n in nobel['Winning Country']:
    if n not in nobel_counted:
        nobel_counted[n] = 1
    else:
        nobel_counted[n] += 1

nobel_countries = list(nobel_counted.keys())
nobel_countries_num = list(nobel_counted.values())
noble_win = pd.DataFrame({"Winning Country": nobel_countries, "Number of times_nob": nobel_countries_num})


#Проверим, все ли страны из итогового датафрейма попали в датафрейм с координатами:
noble_win['Winning Country'].isin(world_new['Winning Country'])

#Часть названий стран - уже неактуальна/по-другому записана в датафрейме с координатами, заменим такие названия вручную:
noble_win.loc[noble_win['Winning Country'] == 'Austria-Hungary', 'Winning Country'] = 'Czech Republic'
noble_win.loc[noble_win['Winning Country'] == 'Soviet Union', 'Winning Country'] = 'Russia'
noble_win.loc[noble_win['Winning Country'] == 'British India', 'Winning Country'] = 'India'
noble_win.loc[noble_win['Winning Country'] == 'West Germany', 'Winning Country'] = 'Germany'
noble_win.loc[noble_win['Winning Country'] == 'United States', 'Winning Country'] = 'United States of America'
noble_win["Winning Country"] = noble_win["Winning Country"].apply(lambda x: x.strip())
noble_win = noble_win.groupby(['Winning Country']).sum().reset_index()
noble_win = noble_win.sort_values(by = 'Number of times_nob', ascending = False)

#Создадим карту стран для лауреатов Нобелевской премии с использованием geopandas:
df_noble_gpd = geopandas.geodataframe.GeoDataFrame(pd.merge(world_new, noble_win, left_on = 'Winning Country', right_on = 'Winning Country'))
df_noble_gpd.explore(column='Number of times_nob', cmap='inferno')


# In[550]:


# Далее рассмотрим географию Букеровской премии.
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
url = 'https://www.bookishelf.com/man-booker-prize-winner/'
site = requests.get(url)
#print(site.status_code) =200 => соединение с сайтом установлено
soup = BeautifulSoup(site.text)
list1 = soup.findAll('h4')

# Введем новую функцию для замены нескольких значений:
def replacer(target_str, to_replace):
    for i, j in to_replace.items():
        target_str = target_str.replace(i, j)
    return target_str
to_replace = {"</h4>,": "$", "by":"", "Book  ":"", "&amp;":"&", "[":"" }
my_str = replacer(str(list1), replace_values)

# В my_str остались вложенные теги. Уберем их с помощью регулярных выражений:
list1_upd = str(re.sub(r'\<[^>]*\>', '', my_str))

#Подготовим данные для создания датафрейма:
list2 = list1_upd.split("$")
del list2[-2]
del list2[-1]
all_names = []
for items in list2:
    p = items.split('\n')
    all_names.append(p)
authors = []
titles = []
counties = []
mistake = []   
        
for j in all_names:
    for i in all_names[all_names.index(j)]:
        if all_names[all_names.index(j)].index(i) == 0:
            titles.append(i)
        elif all_names[all_names.index(j)].index(i) == 1:
            authors.append(i)
        elif all_names[all_names.index(j)].index(i) == 2:
            counties.append(i)
        else:
            mistake.append(i)

#print(len(mistake)) = 0 => все считалось корректно

#Просмотрев все получившиеся списки, видим, что одно из названий выводится с гиперссылкой, исправим это:
titles[42] = ' Schindlers Ark'

df = pd.DataFrame({"Title": titles, "Author": authors, "Country": counties})
#Добавим столбец с годами:
years = ['2022','2021','2020','2019', '2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2002','2001','2000','1999','1998','1997','1996','1995','1994','1993','1992', '1992','1991','1990','1989','1988','1987','1986','1985','1984','1983','1982','1981','1980','1979','1978','1977','1976','1975','1974', '1974','1973','1972','1971','1970', '1970','1969']
df.insert(0, 'Year', years, allow_duplicates = False)
motherland = []
for c in df['Country']:
    if '/' in c:
        b = c.split('/ ')
        motherland.append(b[1])
    else:
        motherland.append(c)
df['Motherland'] = motherland
m_counted = {}
for m in motherland:
    if m not in m_counted:
        m_counted[m] = 1
    else:
        m_counted[m] += 1
booker_countries = list(m_counted.keys())
booker_countries_num = list(m_counted.values())
df_win = pd.DataFrame({"Winning Country": booker_countries, "Number of times_book": booker_countries_num})
df_win = df_win.sort_values(by = 'Number of times_book', ascending = False)


# In[551]:


#Нанесем собранные данные на карту:
world = geopandas.read_file('ne_110m_admin_0_countries.zip')
world_new = world[['SOVEREIGNT','SOV_A3','geometry']]
world_new = world_new.rename(columns={'SOVEREIGNT':"Winning Country"})
df_win["Winning Country"] = df_win["Winning Country"].apply(lambda x: x.strip())
df_win.loc[df_win['Winning Country'] == 'United States', 'Winning Country'] = 'United States of America'
df_booker = pd.merge(world_new, df_win, left_on = 'Winning Country', right_on = 'Winning Country')
#Соединим датафреймы:
df_booker_gpd = geopandas.geodataframe.GeoDataFrame(pd.merge(world_new, df_win, left_on = 'Winning Country', right_on = 'Winning Country'))
df_booker_gpd.explore(column='Number of times_book', cmap="plasma")


# In[552]:


#Далее используем граф для визуализации топ-10 стран по количеству лауреатов каждой из премий, посмотрим на пересечения:
get_ipython().system('pip install networkx')
import networkx as nx
G = nx.Graph()
G.add_node('Booker Prize')
for i in df_win['Winning Country'][:10]:
    G.add_node(i)
    G.add_edge(i, 'Booker Prize')
G.add_node('Noble Prize')
for i in noble_win['Winning Country'][:10]:
    G.add_node(i)
    G.add_edge(i, 'Noble Prize')    
options = {
    'node_color': 'beige',     # цвет узла
    'node_size': 2000,          # размер узла
    'width': 1,                 # ширина соединяющей линии        
    'edge_color':'blue',        # цвет соединяющей линии
    
}

pos2=nx.spring_layout(G,scale=12)
nx.draw(G, pos2, with_labels = True, arrows=False, **options, )


# In[553]:


#Часть 2
#Теперь сравним страны по уровню читательской грамотности: рассмотрим результаты PIRLS - международного исследования качества чтения и понимания текста. 
# Рассмотрим соотношение уровней, полученных на тестировании по читательской грамотности в разных странах.
import pandas as pd
PIRLS = pd.read_csv('PIRLS_2021.csv', delimiter=';', encoding='UTF-8', skiprows=[0,1])
PIRLS.columns = PIRLS.iloc[0]
PIRLS.drop(labels = [0], axis = 0, inplace = True)
PIRLS = PIRLS.rename(columns={'Advanced Benchmark\n(625)':"Advanced", 'High Benchmark \n(550)': 'High', 'Intermediate Benchmark \n(475)': 'Intermediate', 'Low Benchmark \n(400)': 'Low', 'Below Low Benchmark (<400)': 'Below'})
PIRLS = PIRLS.astype({'Advanced': int, 'High': int, 'Intermediate': int, 'Low': int, 'Below':int })
#топ 5 лучших стран по доле школьников с уровнем 'advanced':
PIRLS_adv = PIRLS.sort_values(by = 'Advanced', ascending = False)
PIRLS_adv.head()
PIRLS['Advanced+High'] = PIRLS['Advanced'] + PIRLS['High']
#топ 5 лучших стран по сумме долей школьников с уровнем 'advanced' и 'high':
PIRLS.sort_values(by = 'Advanced+High', ascending = False).head()


# In[554]:


#Визуализируем результаты тестирования PIRLS с помощью библиотеки ipywidgets:
categories=['advance', 'high', 'intermediate', 'low', 'below']
lines = []
with open('PIRLS_2021.csv', encoding='UTF-8') as file:
    for line in file:
        lines.append(line.strip())
countries=[lines[i].split(";")[0] for i in range(8,51)]
d={lines[i].split(";")[0]:[int (j) for j in lines[i].split(";")[1:6]] for i in range(8,51)}

dropdown = Dropdown(options=[(country, d[country]) for country in countries], description='Страна:')

def func(b):
    plt.bar(categories, b['new'], color='cyan');

dropdown.observe(func, names='value')
display(dropdown)


# In[556]:


#Часть 3
#И перейдем к финальной части проекта - анализу пользователей из разных стран на сатей Goodreads.
#На этот раз воспользуемся SQLite.
import csv
import sqlite3
conn = sqlite3.connect('Users1.db')
cur = conn.cursor()
cur.executescript("drop table if exists Users8_info; CREATE TABLE Users8_info (User_ID, Friend_Count, Review_Count, Groups_Count, Favorite_Authors, Joined, Last_Active, Detected_Country, Detected_Gender, About_Character_Length);")
with open('Users.csv','r') as fin:
    dr = csv.DictReader(fin, delimiter=",")
    to_db = [(i['User_ID'], i['Friend_Count'], i['Review_Count'],i['Groups_Count'],i['Favorite_Authors'],i['Joined'],i['Last_Active'],i['Detected Country'],i['Detected Gender'],i['About-Character-Length']) for i in dr]

cur.executemany("INSERT INTO Users8_info (User_ID, Friend_Count, Review_Count, Groups_Count, Favorite_Authors, Joined, Last_Active, Detected_Country, Detected_Gender, About_Character_Length) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
conn.commit()


# In[557]:


#Проверим, что всё считалось корректно:
table1 = pd.read_sql("SELECT * FROM Users8_info;", conn)
table1.head()


# In[558]:


#Исключим из таблицы тех пользователей, у которых не указана страна:
sql_delete_query = """DELETE from Users8_info where Detected_Country = 'None'"""
cur.execute(sql_delete_query)
conn.commit()


# In[559]:


#Создадим новую таблицу для дальнейших исследований:
cur.executescript("""
drop table if exists Users7_info;
create table Users7_info as
select 
    Joined, 
    Last_Active,
    Review_Count,
    Friend_Count
    from 
        Users8_info;
""")
conn.commit()
#Пояснение: Joined - дата, когда пользователь зарегистрировался на сайте; Last_Active - дата, когда пользователь был последний раз активен на сайте; Review_Count - количество отзывов, опубликованных пользователем; Friend_Count - количество друзей на сайте


# In[560]:


#Проверим, что всё считалось корректно:
table2 = pd.read_sql("SELECT * FROM Users7_info;", conn)
table2.head()


# In[561]:


#Для дальнейшего исследования в SQLite попробуем найти время, в течение которого пользователь проявлял активность на сайте. Приведем даннные к типу datetime:
cur.executescript("""
SELECT CAST(Last_Active AS datetime),
       CAST(Joined AS datetime)
from 
        Users8_info;
""")
conn.commit()
conn.close()


# In[562]:


#В ходе многочисленных попыток реализовать задуманное было выяснено, что SQLite не поддерживает ни DATE_TRUNC, ни DATE_DIFF, а также отказывается воспринимать дату в исходном формате с помощью strftime.
#Так было принято решение не изобретать велосипед и воспользоваться datetime в pandos.
table2['Joined'] = pd.to_datetime(table2['Joined'])
table2['Last_Active'] = pd.to_datetime(table2['Last_Active'])
table2['diff_months'] =(table2['Last_Active'] - table2['Joined']) / np.timedelta64 ( 1 , 'M')
#Изменим типы данных и посмотрим коэффициенты корреляции:
table2 = table2.astype({'Review_Count': int, 'Friend_Count': int, 'diff_months': int})
table2.corr()


# In[ ]:


#Есть низкая-средняя корреляция времени активности на сайте и количеством друзей/отзывов в goodreads. Так как новые качественные и подходящие по теме данные найти не представляется возможным на данный момент, построим регрессию с использованием этого датасета. 


# In[563]:


#Сначала построим регрессию, которая отражает зависимость количества друзей в goodreads от страны:
import numpy as np
from sklearn.linear_model import LinearRegression
sales1 = pd.get_dummies (table1, columns=['Detected_Country'], drop_first= False )
x = np.array(sales1.iloc[:, 9:247])
y = np.array(sales1['Friend_Count'])
model1 = LinearRegression().fit(x, y)
r_sq = model1.score(x, y)
print('coefficient of determination:', r_sq)


# In[564]:


#Видим, что R-squared очень низкий - статистически значимой взаимосвязи нет, так как показатель "страна" объясняет вариативность показателя "количество друзей" менее, чем на 1%.
#В таком случае, рассмотрим зависимость количества отзывов от времени, которое пользователь был зарегистрирован на сайте.
a = np.array(table2['diff_months']).reshape(-1, 1)
b = np.array(table2['Review_Count'])
#с = np.array(table2['Friend_Count'])
model2 = LinearRegression().fit(a, b)
r_sq = model2.score(a, b)
print('coefficient of determination:', r_sq)


# In[565]:


#Тоже низкий r^2, но более сильной взаимосвязи между переменными в датасете уже нет, поэтому, продемонстрируем навыки построения регрессий на основе имеющихся данных:
print('intercept:', model2.intercept_)
print('slope:', model2.coef_)


# In[567]:


#Так, мы получили линейную регрессию  b = 3.46 + 1.43*a
#Теперь мы можем предсказывать количество отзывов на Goodreads по времени, которое пользователь был активен на сайте.
#В качестве примера предскажем количество отзывов при времени активности на сайте = 150 месяцев.
b_predicted = model2.intercept_ + model2.coef_ * 150 
print('predicted response:', b_predicted, sep=' ')


# In[ ]:


#На этом всё. Надеюсь, Вам понравилось!

