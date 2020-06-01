import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def pipeline_df(df):
    
    installer_dummies = pd.get_dummies(df['installer'], prefix='installer')
    installer_dummies = installer_dummies[['installer_DWE' , 'installer_Government']]

    funder_dummies = pd.get_dummies(df['funder'], prefix='funder')
    funder_dummies = funder_dummies[['funder_Hesawa' , 'funder_Danida', 'funder_Government Of Tanzania']]

    subvillage_dummies = pd.get_dummies(df['subvillage'], prefix='subvillage')
    subvillage_dummies = subvillage_dummies[['subvillage_Madukani' , 'subvillage_Shuleni', 'subvillage_Majengo']]

    district_dummies = pd.get_dummies(df['district_code'], prefix='district')
    district_dummies = district_dummies[['district_1' , 'district_2', 'district_3', 'district_4']]

    df['construction_year'] = [0 if x == 0 else 1 if x <= 1990 else 2 for x in df['construction_year']]
    year_dummies = pd.get_dummies(df['construction_year'], prefix='construction_year', drop_first=True)

    lga_dummies = pd.get_dummies(df['lga'], prefix='lga')

    lga_dummies = lga_dummies[['lga_Njombe',          
                                'lga_Arusha Rural',
                                'lga_Moshi Rural',   
                                'lga_Bariadi',         
                                'lga_Rungwe',         
                                'lga_Kilosa',         
                                'lga_Kasulu',          
                                'lga_Mbozi',           
                                'lga_Meru',            
                                'lga_Bagamoyo' ]]

    basin_dummies = pd.get_dummies(df['basin'], prefix = 'basin_', drop_first = True)

    region_dummies = pd.get_dummies(df['region'], prefix = 'region_', drop_first = True)

    extraction_dummies = pd.get_dummies(df['extraction_type_class'], prefix = 'extraction_', 
                                        drop_first = True)

    payment_dummies = pd.get_dummies(df['payment'], prefix = 'payment_', drop_first = True)

    quality_dummies = pd.get_dummies(df['quality_group'], prefix = 'quality_', drop_first = True)

    quantity_dummies = pd.get_dummies(df['quantity'], prefix = 'quantity_', drop_first = True)

    source_type_dummies = pd.get_dummies(df['source_type'], prefix = 'source_type', 
                                         drop_first = True)

    source_class_dummies = pd.get_dummies(df['source_class'], prefix = 'source_class', 
                                          drop_first = True)

    waterpoint_type_dummies = pd.get_dummies(df['waterpoint_type_group'], prefix = 'waterpoint_type_',
                                           drop_first = True)

    permit_dict = {True: 1,
                   False: 0}

    df[['permit', 'public_meeting']] = df[['permit', 'public_meeting']].replace(permit_dict)

    df = pd.concat([df, waterpoint_type_dummies, source_class_dummies, source_type_dummies,
                         basin_dummies, region_dummies, extraction_dummies, payment_dummies,
                         quality_dummies, quantity_dummies, funder_dummies, installer_dummies, 
                         subvillage_dummies, district_dummies,lga_dummies, year_dummies], axis = 1)

    df.drop(columns = ['wpt_name', 'date_recorded', 'region_code', 'recorded_by', 
                       'scheme_name', 'extraction_type', 'payment_type', 'water_quality', 
                       'quantity_group', 'source', 'waterpoint_type', 'funder', 'longitude', 
                       'latitude', 'num_private', 'installer', 'subvillage', 'district_code', 
                       'lga', 'scheme_management', 'extraction_type_group', 'construction_year', 
                       'basin', 'region', 'extraction_type_class', 'payment', 'quality_group', 
                       'quantity','source_type', 'source_class', 'waterpoint_type_group'], inplace = True)

    df['ward'] = np.where(df['ward'] == 'Mishamo', 1, 0)
    df['management'] = np.where(df['management'] == 'vwc', 1, 0)
    df['management_group'] = np.where(df['management_group'] == 'user-group', 1, 0)
    
    return df



def pipeline_df_log(df):
    
    installer_dummies = pd.get_dummies(df['installer'], prefix='installer')
    installer_dummies = installer_dummies[['installer_DWE' , 'installer_Government']]
    
    funder_dummies = pd.get_dummies(df['funder'], prefix='funder')
    funder_dummies = funder_dummies[['funder_Hesawa' , 'funder_Danida', 'funder_Government Of Tanzania']]
    
    subvillage_dummies = pd.get_dummies(df['subvillage'], prefix='subvillage')
    subvillage_dummies = subvillage_dummies[['subvillage_Madukani' , 'subvillage_Shuleni', 'subvillage_Majengo']]
    
    district_dummies = pd.get_dummies(df['district_code'], prefix='district')
    district_dummies = district_dummies[['district_1' , 'district_2', 'district_3', 'district_4']]
    
    df['construction_year'] = [0 if x == 0 else 1 if x <= 1990 else 2 for x in df['construction_year']]
    year_dummies = pd.get_dummies(df['construction_year'], prefix='construction_year', drop_first=True)
    
    lga_dummies = pd.get_dummies(df['lga'], prefix='lga')
    lga_dummies = lga_dummies[['lga_Njombe',          
                                'lga_Arusha Rural',
                                'lga_Moshi Rural',   
                                'lga_Bariadi',         
                                'lga_Rungwe',         
                                'lga_Kilosa',         
                                'lga_Kasulu',          
                                'lga_Mbozi',           
                                'lga_Meru',            
                                'lga_Bagamoyo' ]]
    
    basin_dummies = pd.get_dummies(df['basin'], prefix = 'basin_', drop_first = True)
    region_dummies = pd.get_dummies(df['region'], prefix = 'region_', drop_first = True)
    extraction_dummies = pd.get_dummies(df['extraction_type_class'], prefix = 'extraction_', 
                                        drop_first = True)
    payment_dummies = pd.get_dummies(df['payment'], prefix = 'payment_', drop_first = True)
    quality_dummies = pd.get_dummies(df['quality_group'], prefix = 'quality_', drop_first = True)
    quantity_dummies = pd.get_dummies(df['quantity'], prefix = 'quantity_', drop_first = True)
    source_type_dummies = pd.get_dummies(df['source_type'], prefix = 'source_type', 
                                         drop_first = True)
    source_class_dummies = pd.get_dummies(df['source_class'], prefix = 'source_class', 
                                          drop_first = True)
    waterpoint_type_dummies = pd.get_dummies(df['waterpoint_type_group'], prefix = 'waterpoint_type_', drop_first = True)
    
    permit_dict = {True: 1,
                   False: 0}
    
    df[['permit', 'public_meeting']] = df[['permit', 'public_meeting']].replace(permit_dict)
    
    df = pd.concat([df, waterpoint_type_dummies, source_class_dummies, source_type_dummies,
                         basin_dummies, region_dummies, extraction_dummies, payment_dummies,
                         quality_dummies, quantity_dummies, funder_dummies, installer_dummies, 
                         subvillage_dummies, district_dummies,lga_dummies, year_dummies], axis = 1)
    
    df.drop(columns = ['wpt_name', 'date_recorded', 'region_code', 'recorded_by', 
                       'scheme_name', 'extraction_type', 'payment_type', 'water_quality', 
                       'quantity_group', 'source', 'waterpoint_type', 'funder', 'longitude', 
                       'latitude', 'num_private', 'installer', 'subvillage', 'district_code', 
                       'lga', 'scheme_management', 'extraction_type_group', 'construction_year', 
                       'basin', 'region', 'extraction_type_class', 'payment', 'quality_group', 
                       'quantity','source_type', 'source_class', 'waterpoint_type_group'], inplace = True)
    
    df['ward'] = np.where(df['ward'] == 'Mishamo', 1, 0)
    df['management'] = np.where(df['management'] == 'vwc', 1, 0)
    df['management_group'] = np.where(df['management_group'] == 'user-group', 1, 0)
    
    x_permit_test = df[df['permit'].isnull() == True].drop(columns = ['permit', 'public_meeting'], axis=1)
    x_public_test = df[df['public_meeting'].isnull() == True].drop(columns = ['permit', 'public_meeting'], axis=1)
    
    y_permit = df[df['permit'].isnull() == False]['permit']
    y_public = df[df['public_meeting'].isnull() == False]['public_meeting']
    
    x_permit = df[df['permit'].isnull() == False].drop(columns = ['permit', 'public_meeting'], axis=1)
    x_public = df[df['public_meeting'].isnull() == False].drop(columns = ['permit', 'public_meeting'], axis=1)
    
    with open('public_log', 'rb') as handle:
        public_log = pickle.load(handle)
    with open('permit_log', 'rb') as handle:
        permit_log = pickle.load(handle)
        
    public_values = public_log.predict(x_public_test)
    pub_df = pd.DataFrame(public_values)
    x_public_test.reset_index(inplace=True)
    pub_df = pd.concat([x_public_test, pub_df], axis=1)
    pub_df.rename(columns={0: 'public_meeting'}, inplace=True)
    
    permit_values = permit_log.predict(x_permit_test)
    perm_df = pd.DataFrame(permit_values)
    x_permit_test.reset_index(inplace=True)
    perm_df = pd.concat([x_permit_test, perm_df], axis=1)
    perm_df.rename(columns={0:'permit'}, inplace=True)
    
    perm_df = perm_df.set_index('id')['permit']
    pub_df = pub_df.set_index('id')['public_meeting']
    
    df['permit'].fillna(perm_df, inplace = True)
    df['public_meeting'].fillna(pub_df, inplace = True)
    
    return df