
# ================== Import package =====================
import numpy as np
import pandas as pd
# -------------------------------------------------------



def predict_price(nb_acc): 
    tmp_paris_list_data = paris_list_data.copy()
    tmp_paris_list_data['distance'] = tmp_paris_list_data['accommodates'].apply(lambda x: np.abs(x - nb_acc))
    tmp_paris_list_data = tmp_paris_list_data.sort_values('distance')
    nearest_neighbors = tmp_paris_list_data.iloc[0:5]['price']
    predict_price = nearest_neighbors.mean()
    return(predict_price)



if __name__ == '__main__':
    paris_list_data = pd.read_csv('paris_airbnb.csv')
    stripped_comma = paris_list_data['price'].str.replace(',', '')
    stripped_dollars = stripped_comma.str.replace('$', '')
    paris_list_data['price'] = stripped_dollars.astype('float')
    paris_list_data = paris_list_data.loc[np.random.permutation(len(paris_list_data))]
    
    knn_3 = predict_price(3)
    print("Knn with acc 3 ==>",knn_3)
    
    knn_1 = predict_price(1)
    print("Knn with acc 1 ==>", knn_1)
    
    knn_2 = predict_price(2)
    print("Knn with acc 2 ==>", knn_2)        
