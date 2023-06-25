import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict, Counter
from gensim.models import Word2Vec
import folium
from geopy.geocoders import Nominatim
 
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def check_state_california(Latitude, Longitude):
    geolocator = Nominatim(user_agent="geoapiExercises")
 
    location = geolocator.reverse(Latitude+","+Longitude)
    address = location.raw['address']
    # traverse the data
    state = address.get('state', '')
    country = address.get('country', '')

    if state == "California":
        return True, f"You have selected {location}"
    else:
        return False, f"Veuillez entrer une location en Californie, vous avez choisis {location}"
    


def popularity_score(df_resto, m=50):  # m is the number of reviews starting from which the restaurant mean rating becomes more reliable that the global mean rating
    avg_rating = df_resto['avg_rating'].mean()  # mean rating of all restaurants
    # calculate the popularity score for each restaurant
    result = (df_resto['num_of_reviews']/(df_resto['num_of_reviews']+m))*df_resto['avg_rating']\
        + (m/(df_resto['num_of_reviews']+m))*avg_rating
    return result

def get_category_score(df_resto, sim_map):  # sim_map is a dictionary that contains the similarity between each category and the user's preferences
    try:
        return df_resto['category'].apply(lambda x: np.mean([sim_map[c] for c in x]))  # return the mean similarity between the restaurant's categories and the user's preferences
    except:
        return np.zeros(len(df_resto))
@st.cache_data
def preprocess_df_resto(df_resto):
    df_resto['category'] = df_resto['category'].apply(lambda x: x.lower().replace(' ', '').replace('[', '').replace(']', '').replace("'", '').split(','))
    return df_resto
@st.cache_data
def preprocess_df_users(df_users):
    df_users['list_of_resto'] = df_users['list_of_resto'].apply(eval)
    df_users = df_users[df_users['latitude'].notnull()]
    return df_users

def knowledge_score(df_resto, sim_map, user_price, user_lat_norm, user_lng_norm, weights):
    results = df_resto[['gmap_id']].copy()
    results['price_sim'] = 1-df_resto['price'].apply(lambda x: np.abs(x-user_price))/3   # similarity between restaurants prices and the user's preferred price, normalized to [0,1]
    results['category_sim'] = get_category_score(df_resto, sim_map)  # similarity between restaurants categories and the user's preferred categories, normalized to [0,1]
    results['location_sim'] = 1-np.sqrt((df_resto['norm_lat']-user_lat_norm)**2 + (df_resto['norm_lng']-user_lng_norm)**2)  # similarity between restaurants locations and the user's selected location, normalized to [0,1]
    results['score'] = results['price_sim']*weights['price'] + results['category_sim']*weights['category'] + results['location_sim']*weights['location']  # weighted sum of the three similarities
    results['score'] *= 5  # normalize the score to [0,5]
    return results['score']

def content_based_score(df_resto, sim_map, user_price_hist, user_lat_hist, user_lng_hist, weights):
    results = df_resto[['gmap_id']].copy()
    results['price_sim'] = 1-df_resto['price'].apply(lambda x: np.abs(x-user_price_hist))/3  # similarity between restaurants prices and the user's history price, normalized to [0,1]
    results['category_sim'] = get_category_score(df_resto, sim_map)  # similarity between restaurants categories and the user's history categories, normalized to [0,1]
    results['location_sim'] = 1-np.sqrt((df_resto['norm_lat']-user_lat_hist)**2 + (df_resto['norm_lng']-user_lng_hist)**2)  # similarity between restaurants locations and the user's history locations, normalized to [0,1]
    results['score'] = results['price_sim']*weights['price'] + results['category_sim']*weights['category'] + results['location_sim']*weights['location']  # weighted sum of the three similarities
    results['score'] *= 5
    return results['score']

def popularity_rec_weight(n):  # n is the number of interactions of the user with the system
    return 0.95 - 0.9/(1+np.exp(-0.5*n))  # weight attributed to the popularity score given the number of interactions

def knowledge_rec_weight(n):
    return 0.85 - 0.7/(1+np.exp(-0.5*n))  # weight attributed to the knowledge score given the number of interactions

def content_rec_weight(n):
    return 1.6/(1+np.exp(-0.5*n)) - 0.8/(1+np.exp(-0.05*n)) - 0.4  # weight attributed to the content score given the number of interactions

def collab_rec_weight(n):
    return 0.8/(1+np.exp(-0.05*n)) - 0.4  # weight attributed to the collaborative score given the number of interactions

def recommend_to_new_user(df_resto, user_lat, user_lng, user_price, user_categories, model_wv, categories, r=0.05, k=10):  # r is the radius of the circle around the user's location in which the restaurants are selected, and k the number of restaurants to recommend 
    # normalize the user's location
    min_lat, max_lat = df_resto['latitude'].min(), df_resto['latitude'].max()
    min_lng, max_lng = df_resto['longitude'].min(), df_resto['longitude'].max()
    user_lat_norm = (user_lat - min_lat)/(max_lat - min_lat)  # normalize the user's latitude
    user_lng_norm = (user_lng - min_lng)/(max_lng - min_lng)  # normalize the user's longitude
    
    if 'restaurant' in categories:
        categories.remove('restaurant')  # remove the generic category 'restaurant'
    fav_cats = {k: 1/len(user_categories) for k in user_categories}  # normalize the weights to sum to 1
    print(f'User info :\npreferred price: {round(user_price, 2)}\nlocation: ({user_lat}, {user_lng})\npreferred categories: {user_categories}')

    # compute the similarity between the user's favorite categories and the restaurants' categories
    sim_map = defaultdict(lambda: 0)
    sim_map.update({c: np.sum([model_wv.wv.similarity(c, k)*fav_cats[k] for k in fav_cats]) for c in categories if c in model_wv.wv.key_to_index})

    # get the restaurants in the circle around the user's location
    min_nb_resto_around = 100
    df_resto['distance'] = np.sqrt((df_resto['latitude']-user_lat)**2 + (df_resto['longitude']-user_lng)**2)
    if  df_resto[df_resto['distance'] <= r].shape[0] >= min_nb_resto_around:
        df_resto = df_resto[df_resto['distance'] <= r].copy()
    else:
        df_resto = df_resto.sort_values(by='distance', ascending=True).head(min_nb_resto_around).copy()

    weights = {'popu_rec': 0.1, 'knowledge_rec': 0.9}
    
    # compute the score of each restaurant given the recommendation methods
    df_resto['popularity_score'] = popularity_score(df_resto) if weights['popu_rec'] > 0 else 0
    df_resto['knowledge_score'] = knowledge_score(df_resto, sim_map, user_price, user_lat_norm, user_lng_norm, {'price': 0.2, 'location': 0.3, 'category': 0.5}) if weights['knowledge_rec'] > 0 else 0

    # compute the final score of each restaurant with a weighted sum of the scores
    df_resto['score'] = df_resto['popularity_score']*weights['popu_rec'] + df_resto['knowledge_score']*weights['knowledge_rec']
    return df_resto.sort_values(by='score', ascending=False).head(k)  # return the k best restaurants

def content_based_score(df_resto, sim_map, user_price_hist, user_lat_hist, user_lng_hist, weights):
    results = df_resto[['gmap_id']].copy()
    results['price_sim'] = 1-df_resto['price'].apply(lambda x: np.abs(x-user_price_hist))/3
    results['category_sim'] = get_category_score(df_resto, sim_map)
    results['location_sim'] = 1-np.sqrt((df_resto['norm_lat']-user_lat_hist)**2 + (df_resto['norm_lng']-user_lng_hist)**2)
    results['score'] = results['price_sim']*weights['price'] + results['category_sim']*weights['category'] + results['location_sim']*weights['location']
    results['score'] *= 5
    return results['score']

def collab_score(df_resto, user_code, model):
    nb_resto = df_resto.shape[0]
    result = df_resto[['gmap_id']].copy()
    result['score'] = model.predict([np.array([user_code]*nb_resto), df_resto['resto_code']]).reshape(-1)
    return result['score']

def recommend_to_existing_user(df_resto, df_user, user_id, model_ncf, model_wv, categories,  r=0.1, k=10):
    resto_codes = set(range(92363))
    user = df_user[df_user['user_id'] == user_id].iloc[0]
    user_price = user['avg_price']
    user_lat = user['latitude']
    user_lng = user['longitude']
    min_lat, max_lat = df_resto['latitude'].min(), df_resto['latitude'].max()
    min_lng, max_lng = df_resto['longitude'].min(), df_resto['longitude'].max()
    user_lat_norm = (user_lat - min_lat)/(max_lat - min_lat)
    user_lng_norm = (user_lng - min_lng)/(max_lng - min_lng)
    user_code = user['user_code']
    history = user['list_of_resto']
    fav_cats = Counter()
    df_history = df_resto[df_resto['gmap_id'].isin(history)]
    df_history['category'].apply(lambda x: fav_cats.update(Counter(x)))
    if 'restaurant' in fav_cats:
        del fav_cats['restaurant']
    fav_cats = {k:v for k, v in fav_cats.items() if k in sorted(fav_cats, key=fav_cats.get, reverse=True)[:5]}
    print(fav_cats)
    fav_cats = {k: v/sum(fav_cats.values()) for k, v in fav_cats.items()}
    sim_map = defaultdict(lambda: 0)
    sim_map.update({c: np.sum([model_wv.wv.similarity(c, k)*fav_cats[k] for k in fav_cats]) for c in categories if c in model_wv.wv.key_to_index})
    n = len(history)
    weights = {'popu_rec': popularity_rec_weight(n), 'knowledge_rec': knowledge_rec_weight(n), 'content_rec': content_rec_weight(n), 'collab_rec': collab_rec_weight(n)}
    #weights = {'popu_rec': 0, 'knowledge_rec': 0.5, 'content_rec': 0.5, 'collab_rec': 0}
    df_resto['distance'] = np.sqrt((df_resto['latitude']-user_lat)**2 + (df_resto['longitude']-user_lng)**2)
    df_resto = df_resto[(df_resto['distance'] <= r)&(df_resto['resto_code'].isin(resto_codes))].copy()
    print(df_resto.shape)
    df_resto['popularity_score'] = popularity_score(df_resto) if weights['popu_rec'] > 0 else 0
    df_resto['knowledge_score'] = knowledge_score(df_resto, sim_map, user_price, user_lat_norm, user_lng_norm, {'price': 0.2, 'location': 0.3, 'category': 0.5}) if weights['knowledge_rec'] > 0 else 0
    df_resto['content_score'] = content_based_score(df_resto, sim_map, user_price, user_lat_norm, user_lng_norm, {'price': 0.2, 'category': 0.5, 'location': 0.5}) if weights['content_rec'] > 0 else 0
    df_resto['collab_score'] = collab_score(df_resto, user_code, model_ncf) if weights['collab_rec'] > 0 else 0
    df_resto['score'] = df_resto['popularity_score']*weights['popu_rec'] + df_resto['knowledge_score']*weights['knowledge_rec'] + df_resto['content_score']*weights['content_rec'] + df_resto['collab_score']*weights['collab_rec']
    df_history_cats = pd.DataFrame.from_dict(fav_cats, orient='index', columns=['taux d\'occurences'])
    return df_resto.sort_values(by='score', ascending=False).head(k), df_history_cats

@st.cache_data
def load_data(path):
            folder = path
            df_resto = pd.read_csv(folder + 'restaurants.csv')
            df_users = pd.read_csv(folder + 'users.csv')
           
       
            df_resto = preprocess_df_resto(df_resto)
            df_users = preprocess_df_users(df_users)
            return df_resto, df_users
@st.cache_resource
def load_ncfmodel():
            model =tf.keras.models.load_model('./models/ncf_model.h5')
            return model

@st.cache_resource 
def load_word2vecmodel():
            model = Word2Vec.load('./models/word2vec.model')
            return model

def show_results(prediction_score, i,expanded = False):
 
                            
                            expander = st.expander(label =str (i+1)+' ' + prediction_score['name'].iloc[i] ,expanded=expanded)
                                
                            description  = prediction_score['description'].iloc[i]
                            if str(description) == 'nan':    
                                expander.write('**Restaurant**')
                            else:
                                expander.write('**Description**:' + str(description))
                            cat= (prediction_score['category'].iloc[i])
                            expander.write('**Category**:' + str(cat))
                            expander.write('**Price range**: ' + str(prediction_score['price'].iloc[i])+' 💲')
                            expander.write('**Average rating** :'+ str(round(prediction_score['avg_rating'].iloc[i]))+'⭐' )
                                #insert rating as star emoji
                            expander.write('**Address**: '+str(prediction_score['address'].iloc[i]))
                            expander.write('**Google maps url**: '+ str(prediction_score['url'].iloc[i]))

def display_results_on_map(prediction_score,  location):
    
                        newmap = folium.Map(location=location, 
                        zoom_start=8, control_scale=True)

                    
                        for i,row in prediction_score.iterrows():
                            #Setup the content of the popup
                            iframe = folium.IFrame('Name :' + str(row["name"]),height=80,width=120)
                            
                            #Initialise the popup using the iframe
                            popup = folium.Popup(iframe, min_width=300, max_width=300)
                            
                            #Add each row to the map
                            folium.Marker(location=[row['latitude'],row['longitude']],
                                        popup = popup, c=row['name']).add_to(newmap)
                            
                        iframe = folium.IFrame('Utilisateur',height=80,width=120)
                            
                            #Initialise the popup using the iframe
                        popup = folium.Popup(iframe, min_width=300, max_width=300)
                        folium.Marker(location=location,
                                        popup = popup, c=r'Position utilisateur',icon= folium.Icon(color='red')).add_to(newmap)

                        return newmap


def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
