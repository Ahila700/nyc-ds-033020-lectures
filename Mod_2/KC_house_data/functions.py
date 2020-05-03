def water_distances(df, latitudes, longitudes):
    water_dist = []
    for i in range(len(df['lat'])):
        cmin = 100
        for j in range(len(latitudes)):
            c2 = ((df['lat'][i] - latitudes[j])**2 + (df['long'][i] - longitudes[j])**2)**.5
            if c2 < cmin:
                cmin = c2
        water_dist.append(cmin)
    return water_dist

def last_change(df):
    years_since_renovation = []
    for i in range(len(df['yr_built'])):
        if df['yr_renovated'][i] > df['yr_built'][i]:
            years_since_renovation.append(int(df.date[i][0:4]) - df['yr_renovated'][i])
        else:
            years_since_renovation.append(int(df.date[i][0:4]) - df['yr_built'][i])
    return years_since_renovation

def ext_values(df, extreme_cols):
    new_df = df.copy()
    for col in extreme_cols:
        std = new_df[col].std()
        mean = new_df[col].mean()
        value = mean+(5*std)
        new_df[col] = new_df[col].apply(lambda x: value if (x > value) else x)
    return new_df