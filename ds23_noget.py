import streamlit as st
import pandas as pd
import geopy.distance
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap
# import locale
# locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
header = st.container()
dataset = st.container()
features = st.container()
model_training  = st.container()

# st.markdown(
#     """
#     <style>
#     .main {
#     background-color: #F5F5F5;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# @st.cache_data
# def get_data(filename):
#     df = pd.read_excel(filename)
#     return df
file_path_vendas = "vendas.csv" #"data/vendas.csv"
file_path_vendas_predita2 = "vendas_predita2.csv"#"data/vendas_predita2.csv"

with header:
    st.title ('Estudo de viabilidade econômica DS-23')
    st.header('Apresentação')
    st.write("A implantação de uma **dark store** tem se revelado como uma das estratégias mais eficientes adotadas por muitas empresas para atender às necessidades dos clientes em relação à entrega de produtos de forma rápida e segura.")
    st.write("Nesse contexto, a previsão de vendas através de técnicas de machine learning vem sendo utilizada em larga escala como ferramenta indispensável para a tomada de decisões em relação à abertura de novas filiais, especialmente para empresas de comércio eletrônico que buscam atender às demandas dos clientes em relação à entrega de produtos de forma rápida e segura;\n")
    st.write("O presente estudo busca suprir essa busca por informação através do desenvolvimento de modelos estatísticos elaborados a partir da análise de dados históricos de venda, projetando assim uma possível demanda comportamental dos consumidores axiliando a tomada de decisões mais precisas e seguras sobre expansão dos negócios da empresa.")

with dataset:
    st.header('Previsão de Vendas')
    st.write("Apesar de terem sido iniciadas em 2019, o volume de vendas através do comércio eletrônico ainda apresentava uma certa irregularidade, o que poderia vir a prejudicar a homogeneidade dos dados e consequentemente comprometer a análise. Sendo assim, optou-se por utilizar como base de dados as vendas mensais no período de **março/2021** a **fevereiro/2023**.") 
    st.write("Em seguida foram agregadas informações oficiais sobre as características dos municípios de Minas Gerais e Espírito Santo originadas de fontes oficiais do IBGE, CAGED e IPEA, totalizando 252 tipos diferentes de dados quantitativos e qualitativos sobre a população em diversos períodos.")
    st.write("Através o método *StepWise* foram selecionadas cerca de 16 variáveis para treinamento do modelo estatístico relacionadas a:")
    lista = ["Taxa de envelhecimento", "Educação", "Frequência escolar","% População rural", "IDH"]

    for item in lista:
        st.write("- " + item)
    
    
    dtypes = {"cod_IBGE": str, "cidade": str, "UF": str,"latitude": float,"longitude": float,"venda_predita": float }
    df = pd.read_csv(file_path_vendas_predita2,dtype=dtypes,decimal=',',sep=';')
    dtypes = {"Cidade": str, "UF": str,"venda_total": float,"periodo": str  ,"venda_predita": float }
    usecols = ["Cidade", "UF","venda_total",'periodo']
    df_vendas = pd.read_csv(file_path_vendas,dtype=dtypes,usecols=usecols,parse_dates=['periodo'],decimal=',',sep=';')
    df_vendas['Cidade']=df_vendas['Cidade'].str.upper()
    df_soma = df_vendas.groupby('periodo')['venda_total'].sum()
    
    st.markdown('**Vendas mensais utilizadas no período**')
    st.bar_chart(df_soma,use_container_width=True,width=5)
    
    
with features:
    st.header('Metodologia Aplicada')
    st.write('Para o conjunto de dados obtido foi utilizado o algorítmo Random Forest.')
    st.write('O Random Forest é uma técnica de aprendizado de máquina utilizada para prever valores de uma variável alvo - como as vendas, por exemplo - a partir de um conjunto de variáveis explicativas. Essa técnica é baseada em um conjunto de árvores de decisão que são construídas com subconjuntos aleatórios dos dados e variáveis. Cada árvore é treinada em um subconjunto dos dados e, em seguida, a predição final é obtida pela média das predições de todas as árvores.')

with model_training:
    st.header('Simulações')
    st.write('A seguir poderão ser feitas várias simulações de venda em cidades nos estados de **MG** e **ES**.')
    sel_col, disp_col = st.columns(2) 
    # Criar um widget selectbox com as opções de cidades
    # Criar uma lista com as 900 cidades
    cidades = df.cidade.tolist()    
    with sel_col:
        cidade_selecionada = st.selectbox('Selecione uma cidade:', cidades)
        km_dist=sel_col.slider("Qual o raio em km você deseja obter para compor a região da cidade pesquisada?",min_value=10,max_value=200,value=20,step=10)
        
    filtro = (df['cidade'] == cidade_selecionada)
    lat_ref=np.asarray(df[filtro].latitude)[0]
    lon_ref=np.asarray(df[filtro].longitude)[0]
    distances = []
    for lat, lon in zip(df["latitude"], df["longitude"]):
        distance = geopy.distance.distance((lat_ref, lon_ref), (lat, lon)).km
        distances.append(distance)
    df["distancia"] = distances

    df_km = df[(df["distancia"] <= km_dist)&(df["distancia"] >= 0)&(df["cod_IBGE"]!= '313120')].copy()
    df_km['distancia'] = round(df_km['distancia'],0)
    # print(f'\nForam encontradas {df_km.cidade.value_counts().sum()} cidades num raio de {km_dist} km de {cidade_selecionada}')
    soma_vendas = df_km['venda_predita'].sum()
    # valor_formatado = locale.currency(soma_vendas, grouping=True)
    # print('A soma das vendas preditas para a região é de', valor_formatado)
    # st.write(f'**A soma das vendas preditas para a região é de R$ {float(soma_vendas):.2f}**')

    df_pesq=df_km[["cidade", "distancia",'venda_predita']].sort_values('venda_predita',ascending=True)
    df_pesq['venda_predita']=round(df_pesq['venda_predita'],0)
    df_pesq = df_pesq.rename(columns={'venda_predita': 'venda prevista'})
    # df_pesq['venda_predita'] = df_pesq['venda_predita'].apply(lambda x: locale.currency(x, grouping=True))
    total = df_pesq['venda prevista'].sum()
    # total_formatado = locale.currency(total, grouping=True)
    # st.write('Total:', total_formatado)
    # with sel_col:


    with disp_col:        
        st.write(df_pesq.set_index('cidade'))
        
    with sel_col:
        st.markdown(f"**Venda total  projetada: R$ {float(total):.2f}**")
    st.markdown(f"**Mapa de vendas projetadas por região num raio de {km_dist} km**")

    #Criando o objeto do mapa
    mapa2 = folium.Map(location=[lat_ref, lon_ref], zoom_start=10)
    data=df[['latitude', 'longitude', 'venda_predita']].copy()
    ipa_loc=(data[(data['latitude']==-19.7992)&(data['longitude']==-41.7164)].index[0])
    data.drop(ipa_loc,inplace=True)

    # Adicione um mapa de calor com base nos dados de vendas
    heatmap = HeatMap(data=data.values.tolist(),
                      name='Venda projetada',
                      control=True,
                      show=True,
                      min_opacity=0.30,
                      radius=20)
    heatmap.add_to(mapa2)

    # Criar uma função para exibir a soma das vendas previstas nos clusters
    def soma_venda_predita(cluster):
        soma = sum([float(m.get_children()[0].get_tooltip().split(": ")[1].replace(",", ".")) for m in cluster._markers])
        return f'Soma de Vendas Previstas: R$ {soma:,.2f}'

    # Criar um objeto MarkerCluster para agrupar os marcadores próximos
    marker_cluster = MarkerCluster(max_cluster_radius=km_dist)
    
    # Adicionar marcadores personalizados com os valores de 'venda_predita'
    for lat, lon, venda_predita in data.values.tolist():
        folium.Marker(location=[lat, lon], 
                    icon=None,
                    # popup=cidade("Mom & Pop Arrow Shop >>", parse_html=True),
                    tooltip=f'Venda prevista: R$ {float(venda_predita):.2f}').add_to(marker_cluster)
                    

    marker_cluster.add_to(mapa2)

    # Adicionar um controle de camadas para exibir a soma das vendas previstas nos clusters
    folium.map.LayerControl(collapsed=False, overlay=True, control=False, position='topright').add_to(mapa2)
    marker_cluster.add_child(folium.map.Popup(soma_venda_predita))


    # Exibir o mapa com o heatmap e os marcadores no Streamlit
    folium_static(mapa2)

