import streamlit as st
from streamlit_folium import st_folium,folium_static
import folium
from streamlit_option_menu import option_menu
from utils import *


st.set_page_config(page_title='Restaurants Recommendation System', page_icon='🍔', layout='wide', initial_sidebar_state='auto')

local_css("style.css")


#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if "df_resto" and "df_users"  not in st.session_state:
     st.session_state.df_resto, st.session_state.df_users,  = load_data('../data/')

if "ncfmodel" not in st.session_state:
    st.session_state.ncfmodel = load_ncfmodel()

if "word2vecmodel" not in st.session_state:
    st.session_state.word2vecmodel = load_word2vecmodel()

if "all_categories" not in st.session_state:
    all_categories = pd.read_csv('../data/unique_category_updated.csv')
    st.session_state.all_categories = all_categories['category']
     

selected = option_menu(
        menu_title =None,
        options = ['Home','Explorer les tendances', 'Tester utilisateurs existants', 'Nouvel utilisateur'],
        icons= ['house', 'person-lines-fill', 'person-plus'],
        default_index = 0,
        orientation='horizontal',
    )

if selected == 'Home':
    st.session_state.count = 0
    right_column, left_column = st.columns([3,1.2])
    original_title = '<p style="font-family:Impact; color:rgb(238, 78, 78); font-size: 150px; margin-left:20px; margin-top:80px;">RESTAURANT </p>'
    title = '<p style="font-family:Fantasy; font-size: 100px; margin-left: 20px; ">RECOMMENDER</p>'
    right_column.markdown(original_title, unsafe_allow_html=True)
    right_column.markdown(title, unsafe_allow_html=True)
    left_column.image('pizzmm.png',width=600)
    text = '<p style="font-family:serif; font-size: 20px; margin-left: 20px;margin-right:150px;">Bienvenue dans notre application de recommandation de restaurants realisée dans le cadre de notre projet de fin d\'études, ici vous pouvez explorer les tendances du moments, tester les modèles avec des utilisateurs déja inscrits ainsi qu\'intoduire et tester un nouvel utilisateur. </p>'
    right_column.markdown(text, unsafe_allow_html=True)


if selected == 'Explorer les tendances':
    
    st.header('Dans cette section vous pouvez explorer les tendances des restaurants en ce moment')
    #st.selectbox('Sort by',options=['Best rated','Most popular','Price range','Closest'])
   
    
    restos_pop = st.session_state.df_resto.iloc[:200] 
    st.session_state.df_resto['popularity_score'] = popularity_score(st.session_state.df_resto) 
    restos_pop = st.session_state.df_resto.sort_values(by='popularity_score',ascending=False)
    if 'count' not in st.session_state:
        st.session_state.count = 0
   
    
    b1,f,b2 = st.columns([2,15.5,1])
    with b2:
        reset =st.button('Reset')
        if reset:
            
            st.session_state.count =0
            print('rested.?? '+ str(st.session_state.count))
        
    with b1:
        More_button = st.button('Show More')
    
    if More_button:
        print('more')

    placeholder = st.empty()
    placeholder.grid = make_grid(4,4)
    print(st.session_state.count)
    for i in range(2):
                    for j in range(4):
                        with placeholder.grid[i][j]:
                           
                            show_results(restos_pop, st.session_state.count,expanded=True)
                            st.session_state.count+=1

 

if selected == 'Tester utilisateurs existants':
    st.session_state.count = 0
    header = st.container()
    dataset = st.container()
    features = st.container()
    modelTraining = st.container()

    with header:
        st.header('Dans cette section vous pouvez recommander des restaurants à des utilisateurs existants')
 
    with modelTraining:
       
        
        
      
        user_selected= st.selectbox('Choisir un utilisateur',options= st.session_state.df_users['user_id'].head(100).values)
        
        st.write('You selected user:',user_selected)
        Predicted = st.button('Predict')
        print('selected' , user_selected)
        map_l,mi,resto_r = st.columns([2,0.25,2])
        
        
        if Predicted:
            
            resto_r.header('**Top Restaurants**')
            print('Predicted = ')
 
            user_id = user_selected
            prediction_score,history_cats_user = recommend_to_existing_user(st.session_state.df_resto, st.session_state.df_users, user_id, st.session_state.ncfmodel,st.session_state.word2vecmodel,st.session_state.all_categories)
            print(prediction_score)
            for i in range( prediction_score.shape[0]):
                with resto_r:
                    show_results(prediction_score,i)

            
            
            map_l.write('**Map des top restaurants**')
            #Cmap(prediction_score[['latitude','longitude']])
        
            with map_l:
                        map_res = display_results_on_map(prediction_score,  st.session_state.df_users[st.session_state.df_users['user_id'] == user_id][['latitude','longitude']])
                        
                        st_data = folium_static(map_res)
                        st.write('**Historique des catégories préférées de l\'utilisateur:**')
                        
                        st.dataframe(history_cats_user)



if selected == 'Nouvel utilisateur':
    st.session_state.count = 0
    
    header = st.container()
    dataset = st.container()
    features = st.container()
    modelTraining = st.container()

    with header:
        st.header('Avoir une recommandation pour un nouvel utilisateur')
    

    with modelTraining:

 
        if 'session_lat' and 'session_long' not in st.session_state:
            st.session_state.session_lat = 33.69
            st.session_state.session_long = -118.69
        if 'selected_categories'  not in st.session_state:
            st.session_state.selected_categories = ['italianrestaurant']
        
        if 'prediction_score' not in st.session_state:
            st.session_state.prediction_score = pd.DataFrame()

        if 'selected_price' not in st.session_state:
            st.session_state.selected_price = 1
        st.write('**Test a new user:**')
        print('top restart??')
        
        L, M,R = st.columns(([2,0.25,2]))
        with R:
            Latitude,Longitude,selected_categories =33.775541,-118.171894,None

           
            
           
            print('creation map')
            m = folium.Map(location=[36.7783, -119.4179], zoom_start=5)
                    
            m.add_child(folium.LatLngPopup())
            
            f_map = st_folium(m,width = 800, height = 600)
                    
            selected_latitude = Latitude
            selected_longitude = Longitude
                
            if f_map.get("last_clicked"):
                        selected_latitude = f_map["last_clicked"]["lat"]
                        selected_longitude = f_map["last_clicked"]["lng"]

        with L:
            
            prix = st.slider('Select a price level',  1,1, 4)
            st.session_state.selected_price = prix
            st.success(f"Prix selectionné: {st.session_state.selected_price}")
            st.write('Selectionner vos categories preferees')
            selected_categories = st.multiselect(label='Categories',options= st.session_state.all_categories)
            st.write('Clicker sur une location sur la carte pour selectionner la position')
            form = st.form("Position entry form")
           
            submit = form.form_submit_button()
       
            if submit:
                if selected_latitude== Latitude and selected_longitude == Longitude:
                        st.warning("Selected position has default values!")
                        print('warning same loc as default')
                else:
                        st.success(f"Position selectionnée: {selected_latitude}, {selected_longitude}")
             
                st.session_state.session_lat = round(selected_latitude,6)
                st.session_state.session_long = round(selected_longitude,6)
              
                if selected_categories == []:
                      
                        st.warning("Veuillez selectionner une categorie(s)!")
                else:
                      st.session_state.selected_categories = selected_categories
                      st.success(f"categories choisies: {selected_categories}")
                print('stored' + str(st.session_state.session_lat) +' '+ str(st.session_state.session_long))
               
               
            Predicted2 = st.button('Predict')
            
        map_l2,mi2,resto_r2 = st.columns([2,0.25,2])
            
        

        #user_profile = {'latitude': Latitude,'longitude' :Longitude, 'avg_price': prix, 'categories': selected_categories}
        print('start of if predict')
        if Predicted2:
              
                print('saved prof = '+str(st.session_state.session_lat), str(st.session_state.session_long),4,str(st.session_state.selected_categories))
                st.session_state.prediction_score =recommend_to_new_user(st.session_state.df_resto,  st.session_state.session_lat, st.session_state.session_long,st.session_state.selected_price,st.session_state.selected_categories,st.session_state.word2vecmodel,st.session_state.all_categories)
                print('Predicted = ')
                
                print(st.session_state.prediction_score)
                resto_r2.header('**Top Restaurants**')
                
        c1, boo,c2 =  st.columns([2,0.25,2])
                
        if st.session_state.prediction_score.shape[0] == 0:
                    resto_r2.write('No restaurants found nearby')
        else:
                
                with c1:
                        for i in range( st.session_state.prediction_score.shape[0]):
                           
                            show_results( st.session_state.prediction_score, i)
                    
                print('where')
                with c2:
                        map_res = display_results_on_map(st.session_state.prediction_score,[st.session_state.session_lat, st.session_state.session_long])
                        
                    
                        st_data = folium_static(map_res,width=800, height=600)
                        map_l2.write('**Map des top restaurants**')

